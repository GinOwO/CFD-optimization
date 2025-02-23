from datetime import datetime
from pathlib import Path

import numpy as np
from azure.batch import BatchServiceClient, batch_auth
from azure.batch.models import (
    JobAddParameter,
    OutputFile,
    OutputFileBlobContainerDestination,
    OutputFileDestination,
    OutputFileUploadCondition,
    OutputFileUploadOptions,
    ResourceFile,
    TaskAddParameter,
)
from azure.storage.blob import BlobServiceClient
from dotenv import dotenv_values

from azure_utils import get_container_sas_url, upload_files_to_blob
from utils import Parameters

if __name__ == "__main__":
    config = dotenv_values(".env")
    config["_INPUT_CONTAINER_NAME"] += datetime.now().strftime("%Y%m%d%H%M%S")
    config["_JOB_ID"] += datetime.now().strftime("%Y%m%d%H%M%S")
    config["_PROJECT_ROOT_DIR"] = str(Path(config["_PROJECT_ROOT_DIR"]).resolve())

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

    tasks = []

    pop_size = int(config["_POP_SIZE"])
    max_iter = int(config["_MAX_ITER"])

    parameters_str = f'run_name=\\"{run_parameters.run_name}\\",run_path=\\"{run_parameters.run_path}\\",template_path=\\"{run_parameters.template_path}\\",is_debug={run_parameters.is_debug},csv_path=\\"{run_parameters.csv_path}\\",fluid_velocity=np.array({run_parameters.fluid_velocity.tolist()})'
    command_line = f"bash -c \"source /opt/openfoam-dev/etc/bashrc && python3 -c 'from main import *; result = run_distributed(Parameters({parameters_str}), {bounds}, {pop_size}, {max_iter}); print(result)'\""
    print(command_line)

    # Create Blob Storage client
    blob_service_client: BlobServiceClient = BlobServiceClient.from_connection_string(
        f"DefaultEndpointsProtocol=https;AccountName={config['_STORAGE_ACCOUNT_NAME']};AccountKey={config['_STORAGE_ACCOUNT_KEY']};EndpointSuffix=core.windows.net"
    )

    # Create input container and upload files
    input_container_client = blob_service_client.get_container_client(
        config["_INPUT_CONTAINER_NAME"]
    )

    input_container_client.create_container(public_access="blob")

    for file in [
        "main.py",
        "utils.py",
        "cst2coords.py",
        "foil_mesher.py",
        "azure_utils.py",
        ".env",
        # "basic_template.tar.gz",
    ]:
        upload_files_to_blob(
            input_container_client,
            config["_PROJECT_ROOT_DIR"] + f"/{file}",
            prefix="",
        )

    output_container_sas_url = blob_service_client.get_container_client(
        config["_OUTPUT_CONTAINER_NAME"]
    ).url

    # Create Batch client
    credentials = batch_auth.SharedKeyCredentials(
        config["_BATCH_ACCOUNT_NAME"], config["_BATCH_ACCOUNT_KEY"]
    )

    batch_client = BatchServiceClient(
        credentials, batch_url=config["_BATCH_ACCOUNT_URL"]
    )

    # # Create the pool
    # create_batch_pool(batch_client, config)

    # Create the job
    job = JobAddParameter(
        id=config["_JOB_ID"], pool_info={"pool_id": config["_POOL_ID"]}
    )
    batch_client.job.add(job)

    for i in range(int(config["_POOL_NODE_COUNT"])):
        task = TaskAddParameter(
            id=f"task_{i}",
            command_line=command_line,
            resource_files=[
                ResourceFile(
                    auto_storage_container_name="tmpl",
                    blob_prefix="basic_template",
                ),
                ResourceFile(
                    http_url=get_container_sas_url(
                        blob_service_client,
                        config["_INPUT_CONTAINER_NAME"],
                        config,
                    )
                    + "/main.py",
                    file_path="main.py",
                ),
                ResourceFile(
                    http_url=get_container_sas_url(
                        blob_service_client,
                        config["_INPUT_CONTAINER_NAME"],
                        config,
                    )
                    + "/utils.py",
                    file_path="utils.py",
                ),
                ResourceFile(
                    http_url=get_container_sas_url(
                        blob_service_client,
                        config["_INPUT_CONTAINER_NAME"],
                        config,
                    )
                    + "/cst2coords.py",
                    file_path="cst2coords.py",
                ),
                ResourceFile(
                    http_url=get_container_sas_url(
                        blob_service_client,
                        config["_INPUT_CONTAINER_NAME"],
                        config,
                    )
                    + "/foil_mesher.py",
                    file_path="foil_mesher.py",
                ),
                ResourceFile(
                    http_url=get_container_sas_url(
                        blob_service_client,
                        config["_INPUT_CONTAINER_NAME"],
                        config,
                    )
                    + "/.env",
                    file_path=".env",
                ),
            ],
            output_files=[
                OutputFile(
                    file_pattern=f"results*.csv",
                    destination=OutputFileDestination(
                        container=OutputFileBlobContainerDestination(
                            container_url=output_container_sas_url
                        )
                    ),
                    upload_options=OutputFileUploadOptions(
                        upload_condition=OutputFileUploadCondition.task_completion
                    ),
                ),
            ],
        )

        tasks.append(task)

    batch_client.task.add_collection(config["_JOB_ID"], tasks)
