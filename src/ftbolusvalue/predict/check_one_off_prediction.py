"""
    Stub to do one-off prediction using the trained model

    @author: mikhail.galkin
"""

# %% Setup the Tensorflow ------------------------------------------------------
import tensorflow as tf

print("tensorflow ver.:", tf.__version__)


# %% Setup ---------------------------------------------------------------------
import os
import sys
import gzip
import random
import pandas as pd
import numpy as np
from IPython.display import display


# %% Load project's stuff ------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.extend([".", "./.", "././.", "../..", "../../..", "src"])
pd.set_option("display.float_format", lambda x: "%.5f" % x)
from ftbolusvalue.config import LOGGED_MODEL_DIR


# %% Params --------------------------------------------------------------------
RND_STATE = 42
BATCH_SIZE = 2048
N_SAMPLE = 1000

DIR_WITH_DATA = "Z:/fishtailS3/ft-model/bol-us-value/data/"
FILE_WITH_TRAIN_DATA = "bol_us_value_train.csv.gz"
FILE_WITH_TEST_DATA = "bol_us_value_test.csv"
FILEPATH = DIR_WITH_DATA + FILE_WITH_TEST_DATA

COLS_DTYPE = {
    "ade_month": "object",
    "hscode_04": "object",
    "description": "object",
    "weight": "int32",
    "value": "int32",
}
USECOLS = list(COLS_DTYPE.keys())

X_COLS = [
    "ade_month",
    "hscode_04",
    "description",
    "weight",
]
Y_COLS = ["value"]


# %% Custom functions ----------------------------------------------------------
def sample_n_from_csv(
    filepath: str,
    usecols: int = None,
    dtype: dict = None,
    n: int = 1,
    total_rows: int = None,
) -> pd.DataFrame:
    if total_rows is None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                total_rows = sum(1 for row in f)
        except UnicodeDecodeError as e:
            with gzip.open(filepath, "r", encoding="utf-8") as f:
                total_rows = sum(1 for row in f)

    if n > total_rows:
        print("Error: n > total_rows", file=sys.stderr)

    skip_rows = random.sample(range(1, total_rows + 1), total_rows - n)
    df = pd.read_csv(filepath, usecols=usecols, dtype=dtype, skiprows=skip_rows)
    return df


# %% Workflow ------------------------------------------------------------------
def main():
    print(f"\nLoad the 'mg-cargo-value' model ...", end=" ")
    model = tf.keras.models.load_model(LOGGED_MODEL_DIR)
    print("done.")

    print(f"\nGetting random {N_SAMPLE} samples ...", end=" ")
    sample = sample_n_from_csv(FILEPATH, USECOLS, COLS_DTYPE, N_SAMPLE)
    print("done.")

    print(f"Make predictions ...")
    X_ds = tf.data.Dataset.from_tensor_slices(dict(sample[X_COLS])).batch(BATCH_SIZE)
    value_pred = np.exp(model.predict(X_ds))
    value_pred = value_pred.flatten()
    value_true = sample[Y_COLS].values.flatten()

    print(f"Calculate metrics for performance evaluating ...")
    mape = tf.keras.losses.MeanAbsolutePercentageError()
    ttl_mape = mape(value_true, value_pred)
    ttl_diff = 100 * abs((value_true.sum() - value_pred.sum()) / value_true.sum())
    diff = 100 * abs((value_true - value_pred)) / value_true

    display(pd.DataFrame({"true": value_true, "pred": value_pred, "%diff": diff}))
    if N_SAMPLE > 1:
        print(f"MAPE = {ttl_mape}")
        print(f"% diff = {ttl_diff}")


# %% Workflow ===================================================================
if __name__ == "__main__":
    main()
