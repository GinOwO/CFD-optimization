"""
Merged code based on
    https://github.com/NielsBongers/openfoam-airfoil-optimization

Original License: GNU General Public License v3.0 (Same as this repository)
"""

import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import List

import coiled
import numpy as np
from dask.distributed import Client
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from cst2coords import cst2coords
from foil_mesher import meshify
from model import train_models
from utils import FOAM, Parameters, process_result

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)
log.info("Init")

docker_image = "586794457381.dkr.ecr.eu-north-1.amazonaws.com/ginowo/openfoam:latest"
coiled.create_software_environment(
    name="dask-latest",
    container=docker_image,
)


def optimize_surrogate(
    x: np.array,
    models: dict[str : RandomForestClassifier | RandomForestRegressor],
) -> float:
    x = x.reshape(1, -1)

    cls_model: RandomForestClassifier = models["cls_model"]
    reg_model: RandomForestRegressor = models["reg_model"]

    predicted_failure = cls_model.predict(X=x)

    if predicted_failure:
        return np.inf

    parameters = Parameters(
        run_name="svm_optimized_cl_cd",
        run_path=Path("custom_runs"),
        template_path=Path("openfoam_template"),
        is_debug=False,
        csv_path=Path("results/csv/custom_results.csv"),
        fluid_velocity=np.array([99.6194698092, 8.7155742748, 0]),
    )

    case_uuid = str(uuid.uuid4())

    wu = x.squeeze()[0:3]
    wl = x.squeeze()[3:6]

    dz = 0
    N = 50
    run_path = parameters.run_path / case_uuid
    template_path = parameters.template_path

    airfoil_coordinates = cst2coords(wl, wu, dz, N)
    x_vals: List[float] = []
    y_vals: List[float] = []
    for coord in airfoil_coordinates:
        x_vals.append(coord[0])
        y_vals.append(coord[1])
    airfoil_coordinates = np.column_stack((x_vals[:-1], y_vals[:-1]))
    top_section = airfoil_coordinates[1:][0:24]
    bot_section = airfoil_coordinates[1:][25:][::-1]
    top_bottom_difference = top_section[:, 1] - bot_section[:, 1]

    if (top_bottom_difference < 0).any():
        log.info("Airfoil clipping detected")
        return np.inf

    run_path.mkdir(exist_ok=True, parents=True)
    shutil.copytree(src=template_path, dst=run_path, dirs_exist_ok=True)

    FOAM.set_fluid_velocities(run_path, parameters.fluid_velocity)
    meshify(airfoil_coordinates=airfoil_coordinates, case_path=run_path)

    block_mesh_result = FOAM.run_blockmesh(case_path=run_path)
    check_mesh_result = FOAM.run_checkmesh(case_path=run_path)

    if not (block_mesh_result and check_mesh_result):
        log.debug(
            f"Encountered error. Skipping {case_uuid}. blockMesh: {block_mesh_result}. checkMesh: {check_mesh_result}"
        )
        try:
            shutil.rmtree(run_path)
        except Exception:
            pass

        return np.inf

    cl_cd_predicted = reg_model.predict(X=x)

    log.info(f"Model result: {cl_cd_predicted.item()} for {x}")

    try:
        shutil.rmtree(run_path)
    except Exception:
        pass

    return -cl_cd_predicted.item()


def optimize(x: np.array, parameters: Parameters) -> float:
    case_uuid = str(uuid.uuid4()) if not parameters.is_debug else parameters.run_name
    log.info(f"Running case {case_uuid} with {x}")
    wu = x[0:3]
    wl = x[3:6]
    dz = 0
    N = 50
    run_path = Path.cwd() / parameters.run_path / case_uuid
    template_path = Path.cwd() / parameters.template_path
    log.info(f"Run path: {run_path}")
    log.info(f"Template path: {template_path}")
    airfoil_coordinates = cst2coords(wl, wu, dz, N)
    x_vals: List[float] = []
    y_vals: List[float] = []
    for coord in airfoil_coordinates:
        x_vals.append(coord[0])
        y_vals.append(coord[1])
    airfoil_coordinates = np.column_stack((x_vals[:-1], y_vals[:-1]))
    top_section = airfoil_coordinates[1:][0:24]
    bot_section = airfoil_coordinates[1:][25:][::-1]
    top_bottom_difference = top_section[:, 1] - bot_section[:, 1]
    if (top_bottom_difference < 0).any():
        log.info("Airfoil clipping detected")
        process_result(
            x=x_vals,
            parameters=parameters,
            case_uuid=case_uuid,
            run_path=run_path,
            no_clipping=False,
            block_mesh_result=False,
            check_mesh_result=False,
            simple_result=False,
            has_converged=False,
        )
        return np.inf
    run_path.mkdir(exist_ok=True, parents=True)
    if not template_path.exists():
        log.info(f"Template path {template_path} not found.")
        process_result(
            x=x_vals,
            parameters=parameters,
            case_uuid=case_uuid,
            run_path=run_path,
            no_clipping=True,
            block_mesh_result=False,
            check_mesh_result=False,
            simple_result=False,
            has_converged=False,
        )
        return np.inf
    shutil.copytree(src=template_path, dst=run_path, dirs_exist_ok=True)
    FOAM.set_fluid_velocities(run_path, parameters.fluid_velocity)
    meshify(Coords=airfoil_coordinates, run_path=run_path)
    block_mesh_result = FOAM.run_blockmesh(run_path=run_path)
    check_mesh_result = FOAM.run_checkmesh(run_path=run_path)
    if not (block_mesh_result and check_mesh_result):
        log.info(
            f"Encountered error. Skipping {case_uuid}. blockMesh: {block_mesh_result}. checkMesh: {check_mesh_result}"
        )
        process_result(
            x=x_vals,
            parameters=parameters,
            case_uuid=case_uuid,
            run_path=run_path,
            no_clipping=True,
            block_mesh_result=block_mesh_result,
            check_mesh_result=check_mesh_result,
            simple_result=False,
            has_converged=False,
        )
        return np.inf
    with open(run_path / (parameters.run_name + ".foam"), "w") as _:
        pass
    simple_result = FOAM.run_simple(run_path.resolve(), case_uuid)
    if not simple_result:
        log.info(f"Encountered error with SIMPLE. Skipping {case_uuid}.")
        process_result(
            x=x_vals,
            parameters=parameters,
            case_uuid=case_uuid,
            run_path=run_path,
            no_clipping=True,
            block_mesh_result=block_mesh_result,
            check_mesh_result=check_mesh_result,
            simple_result=simple_result,
            has_converged=False,
            cl=np.nan,
            cd=np.nan,
        )
        return np.inf
    df = FOAM.read_force_coefficients(run_path.resolve())
    df["cl_cd_ratio"] = df["Cl"] / df["Cd"]
    converged_std_cl_cd_ratio = df[df["# Time"] > 1500]["cl_cd_ratio"].std()
    if converged_std_cl_cd_ratio > 1.0:
        log.info(f"Solver failed to converge: std = {converged_std_cl_cd_ratio}")
        process_result(
            x=x_vals,
            parameters=parameters,
            case_uuid=case_uuid,
            run_path=run_path,
            no_clipping=True,
            block_mesh_result=block_mesh_result,
            check_mesh_result=check_mesh_result,
            simple_result=simple_result,
            has_converged=False,
            cl=np.nan,
            cd=np.nan,
        )
        return np.inf
    lift_drag_ratio = df["cl_cd_ratio"].iloc[-1]
    log.info(f"Got {lift_drag_ratio}")
    log.info(f"Successfully ran: {case_uuid} - {lift_drag_ratio}")
    if df["Cd"].iloc[-1] < 0:
        log.info(f"Got {df['Cd'].iloc[-1]}: penalizing.")
        process_result(
            x=x_vals,
            parameters=parameters,
            case_uuid=case_uuid,
            run_path=run_path,
            no_clipping=True,
            block_mesh_result=block_mesh_result,
            check_mesh_result=check_mesh_result,
            simple_result=simple_result,
            has_converged=True,
            cl=df["Cl"].iloc[-1],
            cd=df["Cd"].iloc[-1],
        )
        return np.inf
    process_result(
        x=x_vals,
        parameters=parameters,
        case_uuid=case_uuid,
        run_path=run_path,
        no_clipping=True,
        block_mesh_result=block_mesh_result,
        check_mesh_result=check_mesh_result,
        simple_result=simple_result,
        has_converged=True,
        cl=df["Cl"].iloc[-1],
        cd=df["Cd"].iloc[-1],
    )
    return -np.abs(lift_drag_ratio)


def run_distributed(parameters: Parameters, bounds: List):
    seed = 42
    log.info(f"Seed: {seed}")
    log.info(f"Parameters: {parameters}")
    cluster = coiled.Cluster(
        n_workers=3,
        region="eu-north-1",
        worker_memory="8 GiB",
        worker_cpu=4,
        software="dask-latest",
    )
    client = cluster.get_client()
    print("Connected to Coiled cluster at", client.scheduler.address)
    os.environ["DASK_SCHEDULER_ADDRESS"] = client.scheduler.address

    def dask_map(func, iterable):
        futures = client.map(func, iterable)
        return client.gather(futures)

    if len(sys.argv) > 1 and sys.argv[1] == "init":
        result = differential_evolution(
            func=optimize,
            bounds=bounds,
            strategy="best1bin",
            tol=1e-1,
            workers=dask_map,
            popsize=60,
            seed=seed,
            args=(parameters,),
            updating="deferred",
        )
    else:
        cls_model, reg_model = train_models()
        models = {"cls_model": cls_model, "reg_model": reg_model}
        result = differential_evolution(
            optimize_surrogate,
            bounds,
            strategy="best1bin",
            popsize=60,
            tol=1e-1,
            workers=dask_map,
            seed=seed + 1,
            args=(models,),
            updating="deferred",
        )
    log.info(result)

    client.close()
    cluster.close()
    return result


if __name__ == "__main__":
    run_parameters = Parameters(
        run_name="5_degree_AoA_fixed_nu_tilda_reduced_yplus_penalizing_neg_cd_fixed_AoA_angles",
        run_path=Path("openfoam_cases"),
        template_path=Path("basic_template"),
        is_debug=False,
        csv_path=Path("results.csv"),
        fluid_velocity=np.array([99.6194698092, 8.7155742748, 0]),
    )
    bounds = [
        (-1.4400, -0.1027),
        (-1.2552, 1.2923),
        (-0.8296, 0.4836),
        (0.0359, 1.3246),
        (-0.1423, 1.4558),
        (-0.3631, 1.4440),
    ]
    if len(sys.argv) > 1 and sys.argv[1] == "-s":
        run_parameters = Parameters(
            run_name="svm_optimized_cl_cd",
            run_path=Path("openfoam_cases"),
            template_path=Path("basic_template"),
            is_debug=False,
            csv_path=Path("surrogate_results.csv"),
            fluid_velocity=np.array([99.6194698092, 8.7155742748, 0]),
        )
        result = run_distributed(run_parameters, bounds)
    else:
        result = run_distributed(run_parameters, bounds)
    print("Optimization complete. Result:", result)
