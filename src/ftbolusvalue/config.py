from pathlib import Path

# %% Define project's folder paths
PROJECT_PATH = Path(__file__).parent.parent.resolve()
PROJECT_DIR = PROJECT_PATH.parent.resolve()

PROJECT_DATA_DIR = PROJECT_DIR / "data"
PROJECT_DOCS_DIR = PROJECT_DIR / "docs"
PROJECT_REPORTS_DIR = PROJECT_DIR / "reports"
PROJECT_TEMP_DIR = PROJECT_DIR / "temp"
PROJECT_TFBOARD_DIR = PROJECT_DIR / "tensorboard"
PROJECT_TRAKING_DIR = PROJECT_DIR / "tracks"

# %% Define data paths
DATA_DIR_LOCAL = Path("z:/fishtailS3/ft-bol-us-value/data")
DATA_DIR_REMOTE = ""

# %% Training (Trained) model stuff --------------------------------------------
LOGGED_MODEL_DIR = PROJECT_TRAKING_DIR / "20231207-131418/model"

# %% Google Cloud Platform stuff -----------------------------------------------
GCP_LOCATION = ""
GCP_PROJECT_ID = ""
GCP_PROJECT_NAME = ""

GCP_MODEL_BUCKET_NAME = ""
GCP_MODEL_ARTIFACT_URI = ""
GCP_MODEL_DISPLAY_NAME = ""
GCP_MODEL_DESCRIPTION = ""
GCP_MODEL_ENDPOINT_API = ""
GCP_MODEL_ENDPOINT_NAME = ""
GCP_MODEL_SERVING_CONTAINER_URI = ""

# %% Azure Cloud stuff ---------------------------------------------------------
AZURE_SUBSCRIPTION_ID = ""
AZURE_RESOURCE_GROUP = ""
AZURE_WORKSPACE = ""
