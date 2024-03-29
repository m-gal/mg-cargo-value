# %% Setup ---------------------------------------------------------------------
import sys
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


# %% Load project's stuff ------------------------------------------------------
sys.path.extend([".", "./.", "././.", "../..", "../../..", "src"])
from ftbolusvalue.config import AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE


# %%
# from azure.core.exceptions import HttpResponseError

# try:
#     ml_client.compute.get("cpu-cluster")
# except HttpResponseError as error:
#     print("Request failed: {}".format(error.message))

# %% Workflow ------------------------------------------------------------------
# authenticate
credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=AZURE_SUBSCRIPTION_ID,
    resource_group_name=AZURE_RESOURCE_GROUP,
    workspace_name=AZURE_WORKSPACE,
)
