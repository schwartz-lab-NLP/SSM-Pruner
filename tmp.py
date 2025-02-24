from huggingface_hub import snapshot_download, create_repo, upload_folder
import shutil

# Define the source and destination repositories
source_repo = "schwartz-lab/Smol2-Mamba-1.9B"
destination_repo = "schwartz-lab/Smol2-Mamba-1.9B"

# Local directory to temporarily store the downloaded repo
local_dir = "./temp_model_dir"

# Download the source repository, ignoring the 'pytorch_model' directory.
# The glob pattern "pytorch_model/*" excludes all files within that folder.
snapshot_download(
    repo_id=source_repo,
    local_dir=local_dir,
    ignore_patterns=["pytorch_model/*"]
)

# Create the destination repository. Here, we set it to private.
create_repo(repo_id=destination_repo, private=True, exist_ok=True)

# Upload the local folder to the destination repository with a commit message.
upload_folder(
    repo_id=destination_repo,
    folder_path=local_dir,
    commit_message=f"Moved model from {source_repo} to {destination_repo} excluding 'pytorch_model' directory"
)

print(f"Model moved successfully from {source_repo} to {destination_repo}!")

# Optionally, remove the temporary local directory after the upload.
shutil.rmtree(local_dir)