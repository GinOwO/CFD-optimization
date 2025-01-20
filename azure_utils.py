import datetime
import os
from typing import Dict

import numpy as np
from azure.batch import BatchServiceClient
from azure.batch.models import (
    AutoUserScope,
    AutoUserSpecification,
    ContainerConfiguration,
    ElevationLevel,
    ImageReference,
    JobAddParameter,
    OutputFile,
    OutputFileBlobContainerDestination,
    OutputFileUploadCondition,
    OutputFileUploadOptions,
    PoolAddParameter,
    ResourceFile,
    StartTask,
    TaskAddParameter,
    UserIdentity,
    VirtualMachineConfiguration,
)
from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    AccountSasPermissions,
    BlobClient,
    BlobSasPermissions,
    BlobServiceClient,
    ContainerClient,
    ResourceTypes,
    generate_account_sas,
)


def get_container_sas_url(
    blob_service_client: BlobServiceClient,
    container_name: str,
    permissions: BlobSasPermissions,
    config: Dict[str, str],
) -> str:
    sas_token = generate_account_sas(
        account_name=blob_service_client.account_name,
        account_key=config["_STORAGE_ACCOUNT_KEY"],
        resource_types=ResourceTypes(container=True, object=True),
        permission=AccountSasPermissions(
            read=True,
            write=True,
            list=True,
            delete=True,
            add=True,
            create=True,
            update=True,
        ),
        expiry=datetime.datetime.utcnow()
        + datetime.timedelta(hours=int(config["SAS_EXPIRY_HOURS"])),
    )

    # Construct the container SAS URL
    container_sas_url = f"https://{config['_STORAGE_ACCOUNT_NAME']}.blob.core.windows.net/{container_name}?{sas_token}"

    return container_sas_url


def upload_files_to_blob(
    container_client: ContainerClient, local_path: str, prefix: str = ""
):
    """
    Uploads files from a local path to Azure Blob Storage.
    If the local path is a directory, it uploads all files in the directory.
    If the local path is a file, it uploads the single file.
    """
    if os.path.isdir(local_path):
        for root, _, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_path)
                blob_name = os.path.join(prefix, relative_path).replace("\\", "/")
                blob_client = container_client.get_blob_client(blob_name)

                print(f"Uploading {local_file_path} to {blob_name}...")
                with open(local_file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
    elif os.path.isfile(local_path):
        blob_name = os.path.join(prefix, os.path.basename(local_path)).replace(
            "\\", "/"
        )
        blob_client = container_client.get_blob_client(blob_name)

        print(f"Uploading {local_path} to {blob_name}...")
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
    else:
        print(f"Error: Invalid path: {local_path}")


def download_blobs(
    container_client: ContainerClient, local_path: str, prefix: str = ""
):
    """
    Downloads all blobs from the given container to a local directory.

    Args:
        container_client: The Azure Blob Storage container client.
        local_path: The local directory to download files to.
        prefix: An optional prefix to filter the blobs to download.
    """
    print(
        f"Downloading blobs from container: {container_client.container_name} to {local_path} with prefix: {prefix}"
    )

    # Ensure the local directory exists
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # List and download blobs
    blob_list = container_client.list_blobs(name_starts_with=prefix)
    for blob in blob_list:
        print(f"Downloading blob: {blob.name}")

        # Construct the full local file path
        file_path = os.path.join(local_path, os.path.basename(blob.name))

        # Ensure the directory for the file exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Download the blob
        blob_client = container_client.get_blob_client(blob)
        with open(file_path, "wb") as file:
            download_stream = blob_client.download_blob()
            file.write(download_stream.readall())

        print(f"Downloaded {blob.name} to {file_path}")

    print("Blob download complete.")


def create_batch_pool(batch_client, config):
    """
    Creates a new pool inside the Batch account.
    """

    print(f"Creating pool {config['_POOL_ID']}...")
    resource_group_name = config["_RESOURCE_GROUP_NAME"]
    batch_account_name = config["_BATCH_ACCOUNT_NAME"]
    batch_account_url = config["_BATCH_ACCOUNT_URL"]
    pool_id = config["_POOL_ID"]
    vm_size = config["_POOL_VM_SIZE"]
    project_root = config["_PROJECT_ROOT_DIR"]

    # Create a new pool
    new_pool = PoolAddParameter(
        id=pool_id,
        virtual_machine_configuration=VirtualMachineConfiguration(
            image_reference=ImageReference(
                publisher=config["_IMAGE_PUBLISHER"],
                offer=config["_IMAGE_OFFER"],
                sku=config["_IMAGE_SKU"],
                version=config["_IMAGE_VERSION"],
            ),
            node_agent_sku_id=config["_NODE_AGENT_SKU_ID"],
        ),
        vm_size=vm_size,
        target_dedicated_nodes=int(config["_POOL_NODE_COUNT"]),
        start_task=StartTask(
            command_line=f'bash -c "wget {config["_SCRIPT_URL"]} -O install_deps.sh && chmod +x install_deps.sh && ./install_deps.sh"',
            wait_for_success=True,
            user_identity=UserIdentity(
                auto_user=AutoUserSpecification(
                    scope=AutoUserScope.POOL,
                    elevation_level=ElevationLevel.ADMIN,
                )
            ),
            resource_files=[
                ResourceFile(
                    http_url=config["_SCRIPT_URL"],
                    file_path="install_deps.sh",
                )
            ],
        ),
    )
    batch_client.pool.add(new_pool)
    print(f"Pool {config['_POOL_ID']} created.")


def create_batch_job_and_tasks(batch_client, config, parameters, bounds, pop_size):
    """
    Creates a new job and adds tasks to it.
    """

    # Create the job
    job = JobAddParameter(id=config["_JOB_ID"], pool_info=config["_POOL_ID"])
    batch_client.job.add(job)

    # Add tasks to the job
    tasks = []
    for i in range(pop_size):
        # Generate initial population member for this task
        x = np.array([np.random.uniform(low, high) for low, high in bounds])

        # Prepare the task command line
        task_command = (
            f"python3 {config['_MAIN_SCRIPT_PATH']} --x {x} --parameters {parameters}"
        )

        # Add the task
        tasks.append(
            TaskAddParameter(
                id=f"task_{i}",
                command_line=task_command,
            )
        )

    # Add the tasks to the job
    batch_client.task.add_collection(config["_JOB_ID"], tasks)
