import os
import shutil
import uuid
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from distributed import Client
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

    run_path = (
        Path(os.environ["AZ_BATCH_TASK_WORKING_DIR"]) / parameters.run_path / case_uuid
    )  # Make sure OpenFOAM files are created in the working directory
    template_path = (
        Path(os.environ["AZ_BATCH_TASK_WORKING_DIR"]) / parameters.template_path
    )  # Make sure the template is copied to the working directory

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


def run_distributed(scheduler_address, parameters, bounds, pop_size, max_iter):
    """
    Runs the differential evolution optimization in a distributed manner using Dask.
    """

    client = Client(address=scheduler_address)  # Connect to the Dask cluster
    print(f"Connected to Dask scheduler at {scheduler_address}")
    print(client)

    x0 = [
        list(x)
        for x in np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(pop_size, len(bounds)),
        )
    ]

    # Continue with differential_evolution using the best initial population member
    result = differential_evolution(
        optimize,
        bounds,
        strategy="best1bin",
        maxiter=max_iter,
        popsize=1,
        tol=1e-1,
        workers=client.map,
        seed=42,
        args=(parameters,),
        updating="deferred",
        x0=x0,
    )

    client.close()
    print(result)
    return result


if __name__ == "__main__":
    pass
