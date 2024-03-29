from azureml.core import Model
import tensorflow as tf
import json
import numpy as np
import os
import pickle
import joblib
from sklearn.svm import SVC


def init():
    global model
    model_name = "mg-cargo-value"
    path = Model.get_model_path(model_name)
    model = tf.keras.models.load_model(path)


def run(data):
    try:
        data = json.loads(data)

        result = model.predict(data["data"])

        return {"data": result.tolist(), "message": "Successfully classified Iris"}

    except Exception as e:
        error = str(e)

        return {"data": error, "message": "Failed to classify iris"}
