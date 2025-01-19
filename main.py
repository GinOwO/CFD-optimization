import shutil
import uuid
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from cst2coords import cst2coords
from foil_mesher import meshify
from utils import FOAM, Parameters, process_result


def optimize(x: np.array, parameters: Parameters) -> float:
    """Optimization function.

    Args:
        x (np.array): The six CST parameters to generate the airfoil with.
        parameters (Parameters): Settings dataclass.

    Returns:
        float: Goal function performance.
    """

    case_uuid = str(uuid.uuid4()) if not parameters.is_debug else parameters.run_name
    print(f"Running case {case_uuid} with {x}")

    wu = x[0:3]
    wl = x[3:6]
    dz = 0
    N = 50

    run_path = parameters.run_path / case_uuid
    template_path = parameters.template_path

    airfoil_coordinates = cst2coords(wl, wu, dz, N)
    x: List[float] = []
    y: List[float] = []
    for i in range(len(airfoil_coordinates)):
        x.append(airfoil_coordinates[i][0])
        y.append(airfoil_coordinates[i][1])
    airfoil_coordinates = np.column_stack((x[:-1], y[:-1]))

    top_section = airfoil_coordinates[1:][0 : 25 - 1]
    bot_section = airfoil_coordinates[1:][25:][::-1]
    top_bottom_difference = top_section[:, 1] - bot_section[:, 1]

    if (top_bottom_difference < 0).any():
        print("Airfoil clipping detected")
        process_result(
            x=x,
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
    shutil.copytree(src=template_path, dst=run_path, dirs_exist_ok=True)

    FOAM.set_fluid_velocities(run_path, parameters.fluid_velocity)
    meshify(Coords=airfoil_coordinates, run_path=run_path)

    block_mesh_result = FOAM.run_blockmesh(run_path=run_path)
    check_mesh_result = FOAM.run_checkmesh(run_path=run_path)

    if not (block_mesh_result and check_mesh_result):
        print(
            f"Encountered error. Skipping {case_uuid}. blockMesh: {block_mesh_result}. checkMesh: {check_mesh_result}"
        )
        process_result(
            x=x,
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

    with open(
        run_path / (parameters.run_name + ".foam"),
        "w",
    ) as _:
        pass

    simple_result = FOAM.run_simple(run_path.resolve(), case_uuid)

    if not simple_result:
        print(f"Encountered error with SIMPLE. Skipping {case_uuid}.")
        process_result(
            x=x,
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
        print(f"Solver failed to converge: std = {converged_std_cl_cd_ratio}")
        process_result(
            x=x,
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

    print(f"Got {lift_drag_ratio}")
    print(f"Successfully ran: {case_uuid} - {lift_drag_ratio}")

    if df["Cd"].iloc[-1] < 0:
        print(f"Got {df['Cd'].iloc[-1]}: penalizing.")

        process_result(
            x=x,
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
        x=x,
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


def main():
    run_parameters = Parameters(
        run_name="5_degree_AoA_fixed_nu_tilda_reduced_yplus_penalizing_neg_cd_fixed_AoA_angles",
        run_path=Path("openfoam_cases"),
        template_path=Path("basic_template"),
        is_debug=False,
        csv_path=Path("results/csv/results.csv"),
        fluid_velocity=np.array([99.6194698092, 8.7155742748, 0]),
    )

    run_parameters.csv_path.parent.mkdir(exist_ok=True, parents=True)

    bounds = [
        (-1.4400, -0.1027),
        (-1.2552, 1.2923),
        (-0.8296, 0.4836),
        (0.0359, 1.3246),
        (-0.1423, 1.4558),
        (-0.3631, 1.4440),
    ]

    differential_evolution(
        optimize,
        bounds,
        strategy="best1bin",
        maxiter=100000,
        popsize=60,
        tol=1e-1,
        workers=12,
        seed=42,
        args=(run_parameters,),
        updating="deferred",
    )


if __name__ == "__main__":
    main()
