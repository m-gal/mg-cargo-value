""" (mg-cargo-value) ../src/ftbolusvalue/app>uvicorn app:app --reload

Request Body:
    instances = [
        {
            "ade_month": "OCT",
            "hscode_04": "3923",
            "weight": 6240,
            "description": "SPOKE CORE PKGS X X MM SPOKES HS CODE SCAC BANQ HBL SEL FREIGHT COLLECT"
        },
        {
            "ade_month": "APR",
            "hscode_04": "6403",
            "weight": 1083,
            "description": "SPORT SHOES FOOTWEAR LEATHER UPPER MENS CTNS PAI RSGROSS PO NO CUSTOMER ORDER NO EC ARTICLE NO B SIZE QUANTITY HS CODE M ADE IN INDONESIA INV NO NG "
        },
        {
            "ade_month": "FEB",
            "hscode_04": "6105",
            "weight": 919,
            "description": "MEN S RECYCLE POLYESTER KNITTED SWEATSHIRT S C NO HS CODE PO NO CUST NO QTY "
        }
    ]

Response Body:
    {
        "value_usd": [
            [1184.15087890625],
            [9248.1279296875],
            [8119.25830078125]
        ]
    }

True Values:
    "value": 1240,
    "value": 9747,
    "value": 8271,
"""

# %% Standard library imports
import sys, os
import numpy as np

# %% Third party imports
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, RootModel
from typing import List
import tensorflow as tf

# %% Local application imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.extend([".", "./.", "././.", "../..", "../../..", "src"])
# from ftbolusvalue.config import LOGGED_MODEL_DIR

MODEL = "20231207-131418/model"
MODEL_DIR = "D:/dprojects/mgal-for-github/mg-cargo-value/tracks/"
MODEL_PATH = MODEL_DIR + MODEL

VERSION = 1
MODEL = "tensorflow 2.10"
SERVICE = "cargo-import-value"
PATH = f"/api/v{VERSION}"
BATCH_SIZE = 2048

# %% Load model
print(f"\nLoad the 'mg-cargo-value' model ...", end=" ")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except:
    model = tf.keras.models.load_model(MODEL_PATH)
print("done.")


# Initializing a FastAPI App Instance
app = FastAPI()


# Define request body for input data
class Instance(BaseModel):
    ade_month: str
    hscode_04: str
    description: str
    weight: int


class InstanceList(RootModel):
    root: List[Instance]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]


# %%
@app.get(f"{PATH}/import_value")
def root():
    return {"version": VERSION, "model": MODEL, "service": SERVICE}


# Creating an Endpoint to recieve the data to make prediction on.
@app.post(f"{PATH}/import_value/predict")
async def get_prediction(instances: InstanceList):
    """
    FastAPI also validates the request body against the model we have defined
    and returns an appropriate error response.
    """

    input_instances = {
        "ade_month": [],
        "hscode_04": [],
        "description": [],
        "weight": [],
    }

    # Validate data & create inputts for model
    for instance in instances:
        input_instances["ade_month"].append([instance.ade_month])
        input_instances["hscode_04"].append([instance.hscode_04])
        input_instances["description"].append([instance.description])
        input_instances["weight"].append([instance.weight])

    # ! Used for checking prediction in Interactive Mode only
    # for instance in instances:
    #     input_instances["ade_month"].append([instance["ade_month"]])
    #     input_instances["hscode_04"].append([instance["hscode_04"]])
    #     input_instances["description"].append([instance["description"]])
    #     input_instances["weight"].append([instance["weight"]])

    # Cast input data into TensorFlow Dataset
    input_tensor = tf.data.Dataset.from_tensor_slices(input_instances)
    input_tensor = input_tensor.batch(BATCH_SIZE)

    print(f"Make predictions ...", end=" ")
    value_pred = np.exp(model.predict(input_tensor))
    response = {"value_usd": value_pred.tolist()}
    print("Successfully estimated!")

    return JSONResponse(content=response)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
