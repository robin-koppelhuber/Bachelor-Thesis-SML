"""Script to manage artifacts with Weights & Biases

Upload, download, list, and sync artifacts to/from W&B.

Trained models are saved in a persistent cache (shared across runs):
  checkpoints/trained/

Usage:
    # Upload a specific trained model
    python scripts/sync_artifacts.py upload --path checkpoints/trained/chebyshev_abc123.safetensors --type model

    # Upload with custom name and metadata
    python scripts/sync_artifacts.py upload --path model.safetensors --name my-model-v1 --type model \\
        --metadata '{"method": "chebyshev", "epochs": 3}'

    # Download an artifact
    python scripts/sync_artifacts.py download --name my-model-v1:latest --output ./models/

    # List all artifacts in project
    python scripts/sync_artifacts.py list --type model

    # Sync: Upload all trained models
    python scripts/sync_artifacts.py sync --path checkpoints/trained/ --direction upload

    # Sync: Download new/modified W&B artifacts to local cache
    python scripts/sync_artifacts.py sync --path checkpoints/trained/ --direction download

    # Sync: Bidirectional sync (upload + download)
    python scripts/sync_artifacts.py sync --path checkpoints/trained/ --direction both
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of file for tracking"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:8]


def upload_artifact(
    artifact_path: Path,
    artifact_name: str,
    artifact_type: str,
    project: str,
    entity: str = None,
    metadata: dict = None,
    description: str = None,
) -> None:
    """Upload artifact to W&B

    Args:
        artifact_path: Path to file or directory to upload
        artifact_name: Name for the artifact in W&B
        artifact_type: Type of artifact (model, dataset, result, etc.)
        project: W&B project name
        entity: W&B entity (username or team)
        metadata: Optional metadata dictionary
        description: Optional artifact description
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Run: pip install wandb")
        sys.exit(1)

    if not artifact_path.exists():
        logger.error(f"Path does not exist: {artifact_path}")
        sys.exit(1)

    # Initialize W&B
    logger.info(f"Initializing W&B (project: {project}, entity: {entity or 'default'})")
    run = wandb.init(
        project=project,
        entity=entity,
        job_type="upload_artifact",
        name=f"upload_{artifact_name}",
    )

    # Create artifact
    logger.info(f"Creating artifact: {artifact_name} (type: {artifact_type})")
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=description,
        metadata=metadata or {},
    )

    # Add files
    if artifact_path.is_dir():
        logger.info(f"Adding directory: {artifact_path}")
        artifact.add_dir(str(artifact_path))
    else:
        logger.info(f"Adding file: {artifact_path}")
        artifact.add_file(str(artifact_path))

    # Upload
    logger.info(f"Uploading artifact to W&B...")
    run.log_artifact(artifact)

    # Wait for upload to complete
    artifact.wait()

    logger.info(f"✓ Successfully uploaded artifact: {artifact_name}")
    logger.info(f"  View at: {run.get_url()}")

    # Finish run
    run.finish()


def download_artifact(
    artifact_name: str,
    output_path: Path,
    project: str,
    entity: Optional[str] = None,
) -> None:
    """Download artifact from W&B

    Args:
        artifact_name: Artifact name with optional version (e.g., 'model:latest' or 'model:v0')
        output_path: Directory to download artifact to
        project: W&B project name
        entity: W&B entity (username or team)
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Run: pip install wandb")
        sys.exit(1)

    # Initialize W&B API
    logger.info(f"Connecting to W&B (project: {project}, entity: {entity or 'default'})")
    api = wandb.Api()

    # Construct artifact path
    if entity:
        artifact_path = f"{entity}/{project}/{artifact_name}"
    else:
        artifact_path = f"{project}/{artifact_name}"

    # Download artifact
    logger.info(f"Downloading artifact: {artifact_path}")
    try:
        artifact = api.artifact(artifact_path)
        download_dir = artifact.download(root=str(output_path))
        logger.info(f"✓ Successfully downloaded to: {download_dir}")
    except Exception as e:
        logger.error(f"Failed to download artifact: {e}")
        sys.exit(1)


def list_artifacts(
    project: str,
    entity: Optional[str] = None,
    artifact_type: Optional[str] = None,
) -> None:
    """List all artifacts in project

    Args:
        project: W&B project name
        entity: W&B entity (username or team)
        artifact_type: Filter by artifact type
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Run: pip install wandb")
        sys.exit(1)

    # Initialize W&B API
    api = wandb.Api()

    # Construct project path
    if entity:
        project_path = f"{entity}/{project}"
    else:
        project_path = project

    # List artifacts
    logger.info(f"Listing artifacts from {project_path}")
    if artifact_type:
        logger.info(f"  Filtering by type: {artifact_type}")

    try:
        artifacts = api.artifacts(type_name=artifact_type, project_name=project_path)

        print(f"\n{'Name':<40} {'Type':<15} {'Version':<10} {'Size':<10}")
        print("-" * 80)

        for artifact in artifacts:
            size_mb = artifact.size / (1024 * 1024) if artifact.size else 0
            print(f"{artifact.name:<40} {artifact.type:<15} {artifact.version:<10} {size_mb:>8.2f} MB")

    except Exception as e:
        logger.error(f"Failed to list artifacts: {e}")
        sys.exit(1)


def sync_artifacts(
    local_path: Path,
    artifact_type: str,
    project: str,
    entity: Optional[str] = None,
    pattern: str = "*.safetensors",
    direction: str = "upload",
) -> None:
    """Sync local directory with W&B (upload/download/both)

    Args:
        local_path: Local directory to sync
        artifact_type: Artifact type for uploaded files
        project: W&B project name
        entity: W&B entity (username or team)
        pattern: File pattern to match (default: *.safetensors)
        direction: Sync direction (upload, download, both)
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Run: pip install wandb")
        sys.exit(1)

    local_path.mkdir(parents=True, exist_ok=True)

    # Initialize W&B API
    api = wandb.Api()
    if entity:
        project_path = f"{entity}/{project}"
    else:
        project_path = project

    # Get existing artifacts from W&B
    try:
        wandb_artifacts = {a.name: a for a in api.artifacts(type_name=artifact_type, project_name=project_path)}
    except Exception:
        wandb_artifacts = {}

    # Execute sync based on direction
    if direction in ["upload", "both"]:
        _sync_upload(local_path, pattern, wandb_artifacts, artifact_type, project, entity)

    if direction in ["download", "both"]:
        _sync_download(local_path, pattern, wandb_artifacts)


def _sync_upload(
    local_path: Path,
    pattern: str,
    wandb_artifacts: dict,
    artifact_type: str,
    project: str,
    entity: Optional[str],
) -> None:
    """Upload local files that don't exist or changed in W&B"""
    import wandb

    # Find all matching files
    files = list(local_path.glob(pattern))
    if not files:
        logger.info(f"No files matching '{pattern}' found in {local_path}")
        return

    logger.info(f"\n[UPLOAD] Found {len(files)} local file(s) matching '{pattern}'")

    uploaded = 0
    skipped = 0

    # Collect files to upload
    files_to_upload = []
    for file_path in files:
        artifact_name = file_path.stem
        file_hash = get_file_hash(file_path)

        # Check if artifact exists with same content
        if artifact_name in wandb_artifacts:
            existing_artifact = wandb_artifacts[artifact_name]
            existing_hash = existing_artifact.metadata.get('file_hash', None)

            if existing_hash == file_hash:
                logger.info(f"  ⊘ Skip {artifact_name} (unchanged)")
                skipped += 1
                continue

        files_to_upload.append((file_path, artifact_name, file_hash))

    # If nothing to upload, exit early
    if not files_to_upload:
        logger.info(f"[UPLOAD] Complete: {uploaded} uploaded, {skipped} skipped")
        return

    # Create a single wandb run for all uploads
    logger.info(f"Initializing W&B run for batch upload (project: {project}, entity: {entity or 'default'})")
    run = wandb.init(
        project=project,
        entity=entity,
        job_type="sync_artifacts",
        name=f"sync_{len(files_to_upload)}_artifacts",
    )

    try:
        # Upload each artifact using the same run
        for file_path, artifact_name, file_hash in files_to_upload:
            logger.info(f"  ↑ Upload {artifact_name}")

            # Create artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                description=f"Auto-synced from {local_path}",
                metadata={'file_hash': file_hash},
            )

            # Add file
            artifact.add_file(str(file_path))

            # Upload
            run.log_artifact(artifact)

            # Wait for upload to complete
            artifact.wait()

            uploaded += 1
            logger.info(f"    ✓ Successfully uploaded: {artifact_name}")
    finally:
        # Finish the single run
        logger.info("Finishing W&B run")
        run.finish()

    logger.info(f"[UPLOAD] Complete: {uploaded} uploaded, {skipped} skipped")


def _sync_download(
    local_path: Path,
    pattern: str,
    wandb_artifacts: dict,
) -> None:
    """Download W&B artifacts that don't exist locally or changed"""
    import fnmatch

    if not wandb_artifacts:
        logger.info("[DOWNLOAD] No artifacts found in W&B")
        return

    logger.info(f"\n[DOWNLOAD] Found {len(wandb_artifacts)} artifact(s) in W&B")

    # Get local files
    local_files = {f.stem: f for f in local_path.glob(pattern)}

    downloaded = 0
    skipped = 0

    for artifact_name, artifact in wandb_artifacts.items():
        # Check if matches pattern (e.g., *.safetensors -> artifact should end with that)
        if not fnmatch.fnmatch(f"{artifact_name}.safetensors", pattern):
            continue

        # Check if file exists locally with same hash
        if artifact_name in local_files:
            local_file = local_files[artifact_name]
            local_hash = get_file_hash(local_file)
            wandb_hash = artifact.metadata.get('file_hash', None)

            if wandb_hash and local_hash == wandb_hash:
                logger.info(f"  ⊘ Skip {artifact_name} (unchanged)")
                skipped += 1
                continue

        # Download artifact
        logger.info(f"  ↓ Download {artifact_name}")
        try:
            artifact.download(root=str(local_path))
            downloaded += 1
        except Exception as e:
            logger.error(f"  ✗ Failed to download {artifact_name}: {e}")

    logger.info(f"[DOWNLOAD] Complete: {downloaded} downloaded, {skipped} skipped")


def main():
    parser = argparse.ArgumentParser(
        description="Manage artifacts with Weights & Biases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload an artifact")
    upload_parser.add_argument("--path", type=Path, required=True, help="Path to file or directory")
    upload_parser.add_argument("--name", type=str, help="Artifact name (default: filename)")
    upload_parser.add_argument("--type", type=str, default="model", help="Artifact type")
    upload_parser.add_argument("--project", type=str, default="bachelor_thesis_sml", help="W&B project")
    upload_parser.add_argument("--entity", type=str, help="W&B entity")
    upload_parser.add_argument("--metadata", type=str, help='Metadata as JSON')
    upload_parser.add_argument("--description", type=str, help="Artifact description")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download an artifact")
    download_parser.add_argument("--name", type=str, required=True, help="Artifact name (e.g., 'model:latest')")
    download_parser.add_argument("--output", type=Path, default=Path("./downloads"), help="Output directory")
    download_parser.add_argument("--project", type=str, default="bachelor_thesis_sml", help="W&B project")
    download_parser.add_argument("--entity", type=str, help="W&B entity")

    # List command
    list_parser = subparsers.add_parser("list", help="List artifacts")
    list_parser.add_argument("--type", type=str, help="Filter by artifact type")
    list_parser.add_argument("--project", type=str, default="bachelor_thesis_sml", help="W&B project")
    list_parser.add_argument("--entity", type=str, help="W&B entity")

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync local directory with W&B")
    sync_parser.add_argument("--path", type=Path, required=True, help="Local directory to sync")
    sync_parser.add_argument("--type", type=str, default="model", help="Artifact type")
    sync_parser.add_argument("--pattern", type=str, default="*.safetensors", help="File pattern")
    sync_parser.add_argument("--direction", type=str, default="upload", choices=["upload", "download", "both"],
                             help="Sync direction (upload, download, both)")
    sync_parser.add_argument("--project", type=str, default="bachelor_thesis_sml", help="W&B project")
    sync_parser.add_argument("--entity", type=str, help="W&B entity")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "upload":
        metadata = None
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON metadata: {e}")
                sys.exit(1)

        artifact_name = args.name or args.path.stem

        upload_artifact(
            artifact_path=args.path,
            artifact_name=artifact_name,
            artifact_type=args.type,
            project=args.project,
            entity=args.entity,
            metadata=metadata,
            description=args.description,
        )

    elif args.command == "download":
        download_artifact(
            artifact_name=args.name,
            output_path=args.output,
            project=args.project,
            entity=args.entity,
        )

    elif args.command == "list":
        list_artifacts(
            project=args.project,
            entity=args.entity,
            artifact_type=args.type,
        )

    elif args.command == "sync":
        sync_artifacts(
            local_path=args.path,
            artifact_type=args.type,
            project=args.project,
            entity=args.entity,
            pattern=args.pattern,
            direction=args.direction,
        )


if __name__ == "__main__":
    main()
