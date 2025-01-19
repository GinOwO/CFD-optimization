import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class FOAM:
    @staticmethod
    def exec(cmd: str, run_path: Path) -> subprocess.CompletedProcess:
        result = subprocess.run(
            cmd,
            cwd=run_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result

    @staticmethod
    def run_blockmesh(run_path: Path) -> bool:
        result = FOAM.exec(["blockMesh"], run_path)
        return result.returncode == 0

    @staticmethod
    def run_checkmesh(run_path: Path) -> bool:
        result = FOAM.exec(["checkMesh"], run_path)
        if result.returncode == 0:
            mesh_checks_failed = re.search(
                pattern="Failed ([0-9]+) mesh checks", string=result.stdout
            )
            mesh_okay = "Mesh OK." in result.stdout
            if mesh_okay:
                return True
            if mesh_checks_failed:
                return False
        return False

    @staticmethod
    def run_simple(run_path: Path, case_uuid: str) -> bool:
        result = FOAM.exec(["simpleFoam", "-case", run_path], run_path)
        return result.returncode == 0

    @staticmethod
    def set_fluid_velocities(run_path: Path, v: np.array):
        velocity_magnitude = np.linalg.norm(v)
        with open(run_path / "system/controlDict", "r") as f:
            control_dict_template = f.read()

        with open(run_path / "0/U", "r") as f:
            u_template = f.read()

        control_dict_template = control_dict_template.replace(
            "{{v_magnitude}}", str(velocity_magnitude)
        )

        alpha = np.arctan2(v[1], v[0])
        lift_x, lift_y, lift_z = -np.sin(alpha), np.cos(alpha), 0.0
        drag_x, drag_y, drag_z = np.cos(alpha), np.sin(alpha), 0.0

        control_dict_template = control_dict_template.replace("{{lift_x}}", str(lift_x))
        control_dict_template = control_dict_template.replace("{{lift_y}}", str(lift_y))
        control_dict_template = control_dict_template.replace("{{lift_z}}", str(lift_z))
        control_dict_template = control_dict_template.replace("{{drag_x}}", str(drag_x))
        control_dict_template = control_dict_template.replace("{{drag_y}}", str(drag_y))
        control_dict_template = control_dict_template.replace("{{drag_z}}", str(drag_z))

        u_template = u_template.replace("{{v_x}}", str(v[0]))
        u_template = u_template.replace("{{v_y}}", str(v[1]))
        u_template = u_template.replace("{{v_z}}", str(v[2]))

        with open(run_path / "system/controlDict", "w") as f:
            f.write(control_dict_template)

        with open(run_path / "0/U", "w") as f:
            f.write(u_template)

    @staticmethod
    def read_force_coefficients(run_path: Path):
        force_coefficients_path = (
            run_path / "postProcessing/forceCoeffs/0/forceCoeffs.dat"
        )
        df = pd.read_csv(force_coefficients_path, skiprows=8, sep="\t")
        df.columns = [column_name.strip() for column_name in df.columns]
        return df


@dataclass
class Parameters:
    run_name: str
    run_path: Path
    template_path: Path
    is_debug: bool
    csv_path: Path
    fluid_velocity: np.array


def process_result(
    x: np.array,
    parameters: Parameters,
    case_uuid: str,
    run_path: Path,
    no_clipping: bool,
    block_mesh_result: bool,
    check_mesh_result: bool,
    simple_result: bool,
    has_converged: bool,
    cl: Optional[float] = None,
    cd: Optional[float] = None,
):
    """Saving the results to a CSV.

    Args:
        x (np.array): The six CST parameters.
        parameters (Parameters): Settings used during the simulation.
        case_uuid (str): UUID.
        run_path (Path): Path to the case.
        block_mesh_result (bool): Results from blockMesh.
        check_mesh_result (bool): Results from checkMesh.
        simple_result (bool): Result from SIMPLE.
        cl (Optional[float], optional): Coefficient of lift. Defaults to None.
        cd (Optional[float], optional): Coefficient of drag. Defaults to None.
    """
    timestamp = datetime.now().isoformat(timespec="microseconds")

    with open(parameters.csv_path, "a") as f:
        f.write(
            f'{timestamp},{case_uuid},"{parameters.run_name}",{x[0]},{x[1]},{x[2]},{x[3]},{x[4]},{x[5]},{no_clipping},{block_mesh_result},{check_mesh_result},{simple_result},{has_converged},{parameters.fluid_velocity[0]},{parameters.fluid_velocity[1]},{parameters.fluid_velocity[2]},{cl},{cd}\n'
        )
    try:
        shutil.rmtree(run_path) if not parameters.is_debug else None
    except Exception:
        pass
