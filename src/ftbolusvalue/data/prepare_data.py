"""
    Load data from the parquet
    clean & prepare
    split on Train & Test sets

    @author: mikhail.galkin
"""

# %% Setup ---------------------------------------------------------------------
import sys
import pandas as pd
import numpy as np
from sklearn import model_selection
from pathlib import Path
from IPython.display import display


# %% Load project's stuff ------------------------------------------------------
sys.path.extend([".", "./.", "././.", "../..", "../../..", "src"])
from ftbolusvalue.utils.common import outliers_get_zscores
from ftbolusvalue.utils.common import outliers_rid_off
from ftbolusvalue.utils.common import drop_duplicated
from ftbolusvalue.utils.common import pd_set_options


# %% Params --------------------------------------------------------------------
RND_STATE = 42

DIR_WITH_DATA = "Z:/fishtailS3/ft-bol-data/us/ams/data/processed"
FILE_WITH_DATA = "precleaned_all-2016-20.parquet"

DIR_TO_SAVE_DATA = "Z:/fishtailS3/ft-model/bol-us-value/data/"
FILE_TO_SAVE_TRAIN_DATA = "bol_us_value_train.csv.gz"
FILE_TO_SAVE_TEST_DATA = "bol_us_value_test.csv"

PATH_WITH_DATA = Path(DIR_WITH_DATA) / FILE_WITH_DATA
PATH_TO_SAVE_TRAIN_DATA = Path(DIR_TO_SAVE_DATA) / FILE_TO_SAVE_TRAIN_DATA
PATH_TO_SAVE_TEST_DATA = Path(DIR_TO_SAVE_DATA) / FILE_TO_SAVE_TEST_DATA

COLS_TO_READ = [
    "date_estimated_arrival",
    "hschapter",
    "hscode_02",
    "hscode_02_desc_short",
    "hscode_04",
    "hscode_04_desc_short",
    "harmonized_number",
    "harmonized_weight",
    "harmonized_value",
    "port_of_foreign_lading_name",
    "port_of_unlading_name",
    "container_description_text",
]
COLS_NEW_NAMES = {
    "date_estimated_arrival": "ade",
    "harmonized_number": "quantity",
    "harmonized_weight": "weight",
    "harmonized_value": "value",
    "port_of_foreign_lading_name": "port_of_lading",
    "port_of_unlading_name": "port_of_unlading",
    "container_description_text": "description",
}
YEARS_TO_TRAIN = [2016, 2017, 2018, 2019, 2020]

# We use the United States Import Prices from the
# https://tradingeconomics.com/united-states/import-prices
US_IMPORT_PRICES = {
    2016: 120,
    2017: 123,
    2018: 126,
    2019: 126,
    2020: 123,
    2021: 135,
    2022: 145,
}

# %% Custom functions ----------------------------------------------------------
LOWER_LIMIT = 0.01
UPPER_LIMIT = 0.99


def q01(x):
    return x.quantile(0.01)


def q05(x):
    return x.quantile(0.05)


def q50(x):
    return x.quantile(0.5)


def q90(x):
    return x.quantile(0.9)


def q95(x):
    return x.quantile(0.95)


def q99(x):
    return x.quantile(0.99)


def q_lo(x):
    return x.quantile(LOWER_LIMIT)


def q_hi(x):
    return x.quantile(UPPER_LIMIT)


def is_outlier(s, lolim=LOWER_LIMIT, hilim=UPPER_LIMIT, locut=0.01):
    lo = max(s.quantile(lolim), locut)
    hi = s.quantile(hilim)
    return s.between(lo, hi, inclusive="both")


BIN_BOUNDARIES = [
    0,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10_000,
    25_000,
    50_000,
    75_000,
    100_000,
    150_000,
    200_000,
    float("inf"),
]
# %% Read data -----------------------------------------------------------------
pd_set_options()
print(f"Read data from {PATH_WITH_DATA}", end=" ")
df = pd.read_parquet(PATH_WITH_DATA, columns=COLS_TO_READ, engine="fastparquet")
print(f"got {len(df):,} records.")
# display(
#     df.describe(
#         percentiles=[
#             0.01,
#             0.02,
#             0.03,
#             0.04,
#             0.05,
#             0.1,
#             0.25,
#             0.5,
#             0.75,
#             0.9,
#             0.95,
#             0.96,
#             0.97,
#             0.98,
#             0.99,
#             0.995,
#             0.999,
#         ]
#     )
# )

# %% Get the HS Codes section --------------------------------------------------
dfh = pd.read_csv(
    "Z:/fishtailS3/ft-bol-data/us/_webs/hscodes/ft_hscodes_table.csv",
    usecols=["hscode_02", "hssection"],
    dtype={"hscode_02": object},
)
dfh.drop_duplicates(inplace=True)
df = pd.merge(df, dfh, how="left", on="hscode_02")
del dfh


# %% Prepare data --------------------------------------------------------------
print(f"\nSome manipulations with data ...", end=" ")
df = df.rename(columns=COLS_NEW_NAMES)
df = df.dropna(how="any")
df = df.drop_duplicates()
df = df[(df["weight"] > 0) & (df["value"] > 0)]
df = df[df["ade"].dt.year.isin(YEARS_TO_TRAIN)]
df.insert(
    loc=1,
    column="ade_month",
    value=df["ade"].dt.month_name().str[:3].str.upper(),
)
df.insert(
    loc=1,
    column="ade_year",
    value=df["ade"].dt.year,
)
df["value_kg"] = df["value"] / df["weight"]
print(f"remain # {len(df):,} records.")

print(f"Rid off outliers for whole dataest ...")
df_out = outliers_get_zscores(df, ["weight", "value"], sigma=4.5)
df = outliers_rid_off(df,df_out)

print(f"Rid off outliers inside HSCODE-4 groups ...")
print(f"\t< weight > :", end="")
df = df[df.groupby(["hscode_04"])["weight"].transform(is_outlier, locut=10)]
print(f"remain # {len(df):,} records.")
print(f"\t< value > :", end="")
df = df[df.groupby(["hscode_04"])["value"].transform(is_outlier, locut=100)]
print(f"remain # {len(df):,} records.")
print(f"\t< value_kg > :", end="")
df = df[df.groupby(["hscode_04"])["value_kg"].transform(is_outlier, locut=0.05)]
print(f"remain # {len(df):,} records.")

print(f"Rid off outliers inside binned 'weight' groups ...")
df["bin"] = pd.cut(df["weight"], BIN_BOUNDARIES).astype(str)
print(f"\t< bin > :", end="")
df = df[df.groupby(["bin"])["value_kg"].transform(is_outlier, lolim=0.005, hilim=0.995)]
df = df[df.groupby(["bin"])["value"].transform(is_outlier, lolim=0.005, hilim=0.995)]
print(f"remain # {len(df):,} records.")

print(f"Filter out the HSCODE-02 groups with only 1 records ...", end=" ")
df[df.groupby(["hscode_02"]).transform("size") > 1]
print(f"remain # {len(df):,} records.")

print(f"Calculate inflation's indexed values ...")
inflation = {}
for k in US_IMPORT_PRICES.keys():
    inflation[k] = US_IMPORT_PRICES[2022] / US_IMPORT_PRICES[k]
df["value_2022"] = df["value"].mul(df["ade_year"].map(inflation))

df[["quantity", "weight", "value", "value_2022"]] = df[
    ["quantity", "weight", "value", "value_2022"]
].astype("int32")


# %% ---------------------------------------------------------------------------
print(f"< descriptions > & < ports > cleaning with regex ... ")
cols = ["description", "port_of_lading", "port_of_unlading"]
for col in cols:
    print(f"\t< {col} > ... ")
    # First: replace non-words
    df[col] = df[col].str.replace(r"([^\w]|[\d_])+", " ", regex=True)
    # Second: remove redundant whitespaces
    df[col] = df[col].str.replace(r"([\s])+", " ", regex=True)
    df[cols] = df[cols].fillna("[UNK]")


# %% Uppercase -----------------------------------------------------------------
print(f"Uppercasing string columns ...", end=" ")
df = df.apply(lambda col: col.str.upper() if (col.dtype == "object") else col)

df = drop_duplicated(df)


# %% Check processed data ------------------------------------------------------
df["bin"] = pd.cut(df["weight"], BIN_BOUNDARIES)
dfb = df.groupby(["bin"], as_index=False, observed=True).agg(
    {
        "weight": ["size", "mean", "min", "max", q01, q05, q50, q90, q95, q99],
        "value": ["mean", "min", "max", q01, q05, q50, q90, q95, q99],
        "value_kg": ["mean", "min", "max", q01, q05, q50, q90, q95, q99],
    }
)

display(df.describe())
display(dfb.T)


# %% Split processed data ------------------------------------------------------
print(f"\nSplit development data on train & test sets:")
# stratify = df["ade_month"].astype(str) + df["hssection"].astype(str)
df_train, df_test = model_selection.train_test_split(
    df,
    train_size=0.8,
    random_state=RND_STATE,
    shuffle=True,
    stratify=df["hscode_02"],
)
print(f"Dev set: {df.shape[1]} vars & {df.shape[0]:,} rows:")
print(f"\tdf_train: {df_train.shape[1]} vars & {df_train.shape[0]:,} rows,")
print(f"\t\twith a value's total sum = {df_train['value'].sum():,}.")
print(f"\tdf_test: {df_test.shape[1]} vars & {df_test.shape[0]:,} rows,")
print(f"\t\twith a value's total sum = {df_test['value'].sum():,}")


# %% Save processed data -------------------------------------------------------
print(f"Save TRAIN dataset to", end=" ")
path_to_save = DIR_TO_SAVE_DATA + FILE_TO_SAVE_TRAIN_DATA
print(f"{path_to_save}")
df_train.to_csv(
    path_to_save,
    index=False,
    encoding="utf-8-sig",
    compression="gzip",
)

print(f"Save TEST dataset to", end=" ")
path_to_save = DIR_TO_SAVE_DATA + FILE_TO_SAVE_TEST_DATA
print(f"{path_to_save}")
df_test.to_csv(
    path_to_save,
    index=False,
    encoding="utf-8-sig",
)

print("Done.")

# %% RUN.
