""" Reads cleaned PARQUET file US IMPORT
    make prediction for $value

    @author: mikhail.galkin
"""

# %% Setup the Tensorflow ------------------------------------------------------
import tensorflow as tf

print("tensorflow ver.:", tf.__version__)


# %% Setup ---------------------------------------------------------------------
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display


# %% Load project's stuff ------------------------------------------------------
sys.path.extend([".", "./.", "././.", "../..", "../../..", "src"])
pd.set_option("display.float_format", lambda x: "%.5f" % x)
from ftbolusvalue.config import LOGGED_MODEL_DIR


# %% Params --------------------------------------------------------------------
RND_STATE = 42
BATCH_SIZE = 2048

DIR_WITH_DATA = Path("z:/fishtailS3/ft-bol-data/us/xpm/data/processed")
FILE_NAME = "us_xpm_processed"

COLS_INPUT = [
    "ada_month",
    "hscode_04",
    "product_desc",
    "weight_kg_outliers_off",
]
COLS_RENAME = {
    "ada_month": "ade_month",
    "product_desc": "description",
    "weight_kg_outliers_off": "weight",
}
COLS_DTYPE = {
    "ade_month": "object",
    "hscode_04": "object",
    "description": "object",
    "weight": "int32",
    "value": "int32",
}
QUANTILIES = [
    0.001,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.99,
    0.999,
    0.9995,
    0.9999,
]
LO_LIMIT = 0.005
HI_LIMIT = 0.995


# %% Custom functions ----------------------------------------------------------
def replace_outliers(group, lolim=LO_LIMIT, hilim=HI_LIMIT, median=False):
    lo_q = group.quantile(lolim)
    hi_q = group.quantile(hilim)

    lo_out = group < lo_q
    hi_out = group > hi_q

    if median:
        group[lo_out] = group.median()
        group[hi_out] = group.median()
    else:
        group[lo_out] = lo_q
        group[hi_out] = hi_q
    return group


# %% MAIN ----------------------------------------------------------------------
MONTHS = [
    "2019-01",  # 2019
    "2019-02",
    "2019-03",
    "2019-04",
    "2019-05",
    "2019-06",
    "2019-07",
    "2019-08",
    "2019-09",
    "2019-10",
    "2019-11",
    "2019-12",
    "2020-01",  # 2020
    "2020-02",
    "2020-03",
    "2020-04",
    "2020-05",
    "2020-06",
    "2020-07",
    "2020-08",
    "2020-09",
    "2020-10",
    "2020-11",
    "2020-12",
    "2021-01",  # 2021
    "2021-02",
    "2021-03",
    "2021-04",
    "2021-05",
    "2021-06",
    "2021-07",
    "2021-08",
    "2021-09",
    "2021-10",
    "2021-11",
    "2021-12",
    "2022-01",  # 2022
    "2022-02",
    "2022-03",
    "2022-04",
    "2022-05",
    "2022-06",
    "2022-07",
    "2022-08",
    "2022-09",
    "2022-10",
    "2022-11",
    "2022-12",
]

print(f"Load the 'mg-cargo-value' model.")
model = tf.keras.models.load_model(LOGGED_MODEL_DIR)
display(model.summary())

for m in MONTHS:
    month = m.replace("-", "")
    path = Path(f"{DIR_WITH_DATA}/{FILE_NAME}_{month}.parquet")
    print(f"\nReading a data for '{m}' from {str(path)} ...")
    df = pd.read_parquet(path, engine="fastparquet")
    print(f"Got the {len(df):,} records & {df.shape[1]} columns.")

    print(f"Prepare data for prediction ...")
    df_input = df[COLS_INPUT].copy()
    df_input.rename(columns=COLS_RENAME, inplace=True)
    df_input["ade_month"] = df_input["ade_month"].dt.month_name().str[:3].str.upper()
    df_input["weight"] = df_input["weight"].astype("int64")
    # First: replace non-words
    df_input["description"] = df_input["description"].str.replace(
        r"([^\w]|[\d_])+",
        " ",
        regex=True,
    )
    # Second: remove redundant whitespaces
    df_input["description"] = df_input["description"].str.replace(
        r"([\s])+",
        " ",
        regex=True,
    )
    df_input[["hscode_04", "description"]] = df_input[["hscode_04", "description"]].fillna("[UNK]")

    # # * OPTIONAL : Get the HS Codes descriptions ('hssection')
    # print(f"Add the 'hssection' column ...")
    # dfh = pd.read_csv(
    #     "Z:/fishtailS3/ft-bol-data/us/_webs/hscodes/ft_hscodes_table.csv",
    #     usecols=["hscode_02", "hssection"],
    #     dtype={"hscode_02": object},
    # )
    # dfh.drop_duplicates(inplace=True)
    # df = pd.merge(df, dfh, how="left", on="hscode_02")
    # df = df[sorted(df.columns)]

    print(f"Make predictions ...")
    ds_input = tf.data.Dataset.from_tensor_slices(dict(df_input))
    ds_input = ds_input.batch(BATCH_SIZE)
    value_pred = np.exp(model.predict(ds_input))
    value_pred = value_pred.flatten()

    # Check predictions
    zeros = np.count_nonzero(value_pred == 1)
    print(f"Zeros predicted: # {zeros:,} - % {zeros / len(value_pred) * 100}")
    print(f"Previous total sum $USD: {df['value_usd_pred'].sum():,}")
    print(f"Evaluated total sum $USD: {value_pred.sum():,}")
    print(f"Predicted minimum: {value_pred.min()}")
    print(f"Predicted maximum: {value_pred.max():,}")
    qq = pd.DataFrame(
        {
            "value_pred": value_pred,
            "value_prev": df["value_usd_pred"],
            "usd_kg_pred": value_pred / df["weight_kg_outliers_off"],
            "usd_kg_prev": df["usd_kg"],
        }
    )
    display(qq.describe(QUANTILIES))

    # The dtype of prediction=float32. When we convert it to Pandas float
    # the total summ will slightly changed.
    df["value_usd_pred"] = value_pred
    df["value_usd_pred"] = df["value_usd_pred"].astype(float).round(0)
    df["usd_kg"] = df["value_usd_pred"] / df["weight_kg_outliers_off"]
    print(f"Final evaluated total sum $USD: {df['value_usd_pred'].sum():,}")

    # # print(f"Replace outliers in predicted value ...")
    # # print(f"Evaluated total sum $USD before: {df['value_usd_pred'].sum():,}")
    # # print(f"Stats for USD\kg:")
    # # display(df["usd_kg"].describe(QUANTILIES))

    # # df["usd_kg"] = df.groupby(["hscode_04"])["usd_kg"].transform(replace_outliers)
    # # df["value_usd_pred"] = df["usd_kg"] * df["weight_kg_outliers_off"]
    # # print(f"Evaluated total sum $USD after: {df['value_usd_pred'].sum():,}")
    # # print(f"Stats for USD\kg:")
    # # display(df["usd_kg"].describe(QUANTILIES))

    # # # ! In case when outliers still exists
    # # param = df["value_usd_pred"].sum()
    # # while param > 100_000_000_000:
    # #     df["usd_kg"] = df.groupby(["hscode_04"])["usd_kg"].transform(
    # #         replace_outliers,
    # #         median=True,
    # #     )
    # #     df["value_usd_pred"] = df["usd_kg"] * df["weight_kg_outliers_off"]
    # #     print(f"Evaluated total sum $USD after: {df['value_usd_pred'].sum():,}")
    # #     param = df["value_usd_pred"].sum()

    # # print(f"Final stats for USD\kg:")
    # # display(df["usd_kg"].describe(QUANTILIES))

    path = Path(f"{DIR_WITH_DATA}/{FILE_NAME}_{month}_val.parquet")
    print(f"Saving as {path} ...")
    df.to_parquet(
        path,
        engine="auto",
        compression="snappy",
        index=False,
    )
    print("Done.")
