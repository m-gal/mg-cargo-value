"""
    General Multi-Layer Perceptron NN model.

    @author: mikhail.galkin
"""

#### WARNING: the code that follows would make you cry.
#### A Safety Pig is provided below for your benefit. ¯\_(ツ)_/¯
#                            _
#    _._ _..._ .-',     _.._(`))
#   '-. `     '  /-._.-'    ',/
#      )         \            '.
#     / _    _    |             \
#    |  a    a    /              |
#    \   .-.                     ;
#     '-('' ).-'       ,'       ;
#        '-;           |      .'
#           \           \    /
#           | 7  .__  _.-\   \
#           | |  |  ``/  /`  /
#          /,_|  |   /,_/   /
#            /,_/      '`-'


# %% Setup the Tensorflow ------------------------------------------------------
import tensorflow as tf

print("tensorflow ver.:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        cuda = tf.test.is_built_with_cuda()
        gpu_support = tf.test.is_built_with_gpu_support()
        print(f"\tPhysical GPUs: {len(gpus)}\n\tLogical GPUs: {len(logical_gpus)}")
        print(f"\tIs built with GPU support: {gpu_support}")
        print(f"\tIs built with CUDA: {cuda}")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# %% Setup ---------------------------------------------------------------------
import sys
import json
import inspect
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import model_selection
from pprint import pprint
from IPython.display import display
import re
import string

# %% Load project's stuff ------------------------------------------------------
sys.path.extend([".", "./.", "././.", "../..", "../../..", "src"])

from ftbolusvalue.config import PROJECT_TRAKING_DIR
from ftbolusvalue.utils.dl import df_to_ds
from ftbolusvalue.utils.dl import create_model_inputs_and_features
from ftbolusvalue.utils.dl import plot_history
from ftbolusvalue.utils.dl import plot_pediction_errors
from ftbolusvalue.utils.common import drop_duplicated
from ftbolusvalue.utils.common import pd_set_options


# %% Params --------------------------------------------------------------------
RND_STATE = 42

DIR_WITH_DATA = "Z:/fishtailS3/ft-model/bol-us-value/data/"
FILE_WITH_TRAIN_DATA = "bol_us_value_train.csv.gz"
FILE_WITH_TEST_DATA = "bol_us_value_test.csv"

COLS_DTYPE = {
    "ade_month": "object",
    "hscode_04": "object",
    "description": "object",
    # "port_of_lading": "object",
    # "port_of_unlading": "object",
    "weight": "int64",
    "value": "int64",
}
USECOLS = list(COLS_DTYPE.keys())

X_COLS = [
    "ade_month",
    "hscode_04",
    "description",
    # "port_of_lading",
    # "port_of_unlading",
    "weight",
]
Y_COLS = ["value"]

MODEL_PARAMS = {
    "_batch_size": 4096,
    "_epochs": 24,
    "_optimizer": tf.keras.optimizers.RMSprop(learning_rate=0.001),
    # "_optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
    # "_optimizer": tf.keras.optimizers.Ftrl(learning_rate=0.001),
    # "_loss": tf.keras.losses.MeanSquaredError(name="mse"),
    "_loss": tf.keras.losses.MeanAbsoluteError(name="mae"),  # +
    # "_loss": tf.keras.losses.MeanAbsolutePercentageError(name="mape"),
    # "_loss": tf.keras.losses.MeanSquaredLogarithmicError(name="msle"),
    "_metrics": tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
}

# To get reproducible results
np.random.seed(RND_STATE)
tf.keras.utils.set_random_seed(RND_STATE)
pd_set_options()


# %% Load processed data -------------------------------------------------------
print(f"Load DEV dataset from", end=" ")
path_to_data = DIR_WITH_DATA + FILE_WITH_TRAIN_DATA
print(f"{path_to_data}")
df_train = pd.read_csv(path_to_data, dtype=COLS_DTYPE, usecols=USECOLS)
print(f"Loaded dataset has {df_train.shape[0]:,} rows and {df_train.shape[1]} cols.")

print(f"Load TEST dataset from", end=" ")
path_to_data = DIR_WITH_DATA + FILE_WITH_TEST_DATA
print(f"{path_to_data}")
df_test = pd.read_csv(path_to_data, dtype=COLS_DTYPE, usecols=USECOLS)
print(f"Loaded dataset has {df_test.shape[0]:,} rows and {df_test.shape[1]} cols.")

# print(f"\nRemove duplicates:")
# df_train = drop_duplicated(df_train)
# df_test = drop_duplicated(df_test)


# %% Slit DEV set onto TRAIN and VAL -------------------------------------------
# Split the development dataset with stratified target
print(f"\nSplit development data on train & validation sets ...")
df_train, df_val = model_selection.train_test_split(
    df_train,
    train_size=0.8,
    random_state=RND_STATE,
    shuffle=True,
    stratify=df_train["ade_month"],
)

print(f"\tdf_train: {df_train.shape[1]} vars & {df_train.shape[0]:,} rows.")
print(f"\tdf_val: {df_val.shape[1]} vars & {df_val.shape[0]:,} rows.")
print(f"\tdf_test: {df_test.shape[1]} vars & {df_test.shape[0]:,} rows.")


# %% Handle target -------------------------------------------------------------
# Save separatedly the true target values
Y_true_train = df_train[Y_COLS].values.astype("int64")
Y_true_val = df_val[Y_COLS].values.astype("int64")
Y_true_test = df_test[Y_COLS].values.astype("int64")

# Log transformation for target values
# astype("float32") helpful to decrease memory used
df_train[Y_COLS] = np.log(df_train[Y_COLS])  # .astype("float32")
df_val[Y_COLS] = np.log(df_val[Y_COLS])  # .astype("float32")
df_test[Y_COLS] = np.log(df_test[Y_COLS])  # .astype("float32")


# %% Convert pd.DataFrame to tf.Dataset ----------------------------------------
#! Do not use the shuffle=True! It leads to wrong accuracy on the test set
#! also a runtime increases x10 times
print("Train set:", end=" ")
ds_train, y_train_true = df_to_ds(
    df_train,
    X_COLS,
    Y_COLS,
    MODEL_PARAMS["_batch_size"],
    shuffle=False,
)
print("Val set:", end=" ")
ds_val, y_val_true = df_to_ds(
    df_val,
    X_COLS,
    Y_COLS,
    MODEL_PARAMS["_batch_size"],
    shuffle=False,
)
print("Test set:", end=" ")
ds_test, y_test_true = df_to_ds(
    df_test,
    X_COLS,
    Y_COLS,
    MODEL_PARAMS["_batch_size"],
    shuffle=False,
)

print(f"\nInspect the train dataset's elements ...")
pprint(ds_train.element_spec)


# %% Setup a FeatureSpace ------------------------------------------------------
def custom_standardizer(string_tensor):
    string_tensor = tf.strings.lower(string_tensor)
    string_tensor = tf.strings.regex_replace(
        string_tensor, f"[{re.escape(string.punctuation)}]", " "
    )
    string_tensor = tf.strings.regex_replace(string_tensor, "([^\w]|[\d_])+", " ")
    string_tensor = tf.strings.regex_replace(string_tensor, "([\s])+", " ")
    return string_tensor


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
    100_000,
    float("inf"),
]

# A lists of the binary feature names. Feature like "is_animal"
BINARY_FEATURES = []  # ["is_animal"]

# A dictionaries of the features' names and **kwargs for preprocessing layers.
NUMERIC_TO_NORMALIZED = {
    None: {
        "axis": -1,
        "mean": None,
        "variance": None,
    }
}
NUMERIC_TO_DISCRETIZED = {
    "weight": {
        "bin_boundaries": None,
        "num_bins": 32,  # max = 100
        "output_mode": "int",  # (!"int", "one_hot", "multi_hot", "count")
    }
}

# Categorical features w/o vocabulary we want be encoding.
CATEGORICAL_INTEGER_TO_ENCODED = {
    None: {
        # * Related to Lookup layer:
        "max_tokens": None,
        "num_oov_indices": 1,
        "mask_token": None,
        "oov_token": -1,
        "vocabulary": None,
        "output_mode": "one_hot",  # (!"int", "one_hot", "multi_hot", "count", "tf_idf")
    }
}
CATEGORICAL_STRING_TO_ENCODED = {
    "ade_month": {
        # * Related to Lookup layer:
        "max_tokens": None,
        "num_oov_indices": 1,
        "mask_token": None,
        "oov_token": "[UNK]",
        "vocabulary": None,
        "output_mode": "one_hot",  # (!"int", "one_hot", "multi_hot", "count", "tf_idf")
    },
    # "port_of_lading": {
    #     # * Related to Lookup layer:
    #     "max_tokens": None,
    #     "num_oov_indices": 1,
    #     "mask_token": None,
    #     "oov_token": "[UNK]",
    #     "vocabulary": None,
    #     "output_mode": "one_hot",  # (!"int", "one_hot", "multi_hot", "count", "tf_idf")
    # },
    # "port_of_unlading": {
    #     # * Related to Lookup layer:
    #     "max_tokens": None,
    #     "num_oov_indices": 1,
    #     "mask_token": None,
    #     "oov_token": "[UNK]",
    #     "vocabulary": None,
    #     "output_mode": "one_hot",  # (!"int", "one_hot", "multi_hot", "count", "tf_idf")
    # },
}

# Categorical features with vocabulary we want be one-hot encoding.
CATEGORICAL_INTEGER_TO_EMBEDDED = {
    None: {
        "use_embedding": True,
        # * Related to Lookup layer:
        "max_tokens": None,
        "num_oov_indices": 1,
        "mask_token": None,
        "oov_token": -1,
        "vocabulary": None,
        "output_mode": "int",  # (!"int", "one_hot", "multi_hot", "count", "tf_idf")
        # * Related to Embedding layer:
        "output_dim": None,
        "mask_zero": False,
        "input_length": None,
    }
}
CATEGORICAL_STRING_TO_EMBEDDED = {
    "hscode_04": {
        "use_embedding": False,
        # * Related to Lookup layer:
        "max_tokens": None,
        "num_oov_indices": 1,
        "mask_token": None,
        "oov_token": "[UNK]",
        "vocabulary": sorted(df_train["hscode_04"].unique().tolist()),
        "output_mode": "multi_hot",  # (!"int", "one_hot", "multi_hot", "count", "tf_idf")
        # * Related to Embedding layer:
        "output_dim": None,
        "mask_zero": False,
        "input_length": None,
    }
}

# A list of textual features w/o vocabulary we want be embedded
TEXT_TO_EMBEDDED = {
    "description": {
        "use_embedding": False,
        # * Related to TextVectorization layer:
        "max_tokens": 20_000,  # 20_000 good choice for big text - Not necessary
        "standardize": "lower_and_strip_punctuation",
        "split": "whitespace",  # (None, "character")
        "ngrams": 2,
        "output_mode": "multi_hot",  # (!"int", "multi_hot", "count", "tf_idf")
        "output_sequence_length": None,  # Only valid in "int" mode
        "pad_to_max_tokens": False,  # Only valid in "multi_hot", "count", "tf_idf"
        "vocabulary": None,
        "idf_weights": None,  # Only valid in "tf_idf" mode
        "sparse": False,  # Bool. Only applicable to "int" mode
        "ragged": False,  # Bool. Only applicable to "multi_hot", "count", "tf_idf" modes
        # * Related to Embedding layer:
        "output_dim": None,  # None or Int
        "mask_zero": False,  # Bool.
        "input_length": None,  # None or Int
        "flat_or_pool": "pool",  # ("pool", "flat"). To use Flatten or GlobalAveragePooling1D
    }
}

# Targets' names and types
REGRESSION_TARGETS = ["value"]
BINARY_TARGETS = []
MULTICLASS_TARGET_W_CLASSES = {}  # {MULTICLASS_TARGET_NAME: classes}
MULTILABEL_TARGET_W_LABELS = {}  # {MULTILABEL_TARGET_NAME: labels}


# A dictionary of all the input columns\features.
set_of_features = {
    "x": X_COLS,
    "y": Y_COLS,
    "x_type": {
        "binary": BINARY_FEATURES,
        "numeric_to_normalized": NUMERIC_TO_NORMALIZED,
        "numeric_to_discretized": NUMERIC_TO_DISCRETIZED,
        "categorical_int_to_encoded": CATEGORICAL_INTEGER_TO_ENCODED,
        "categorical_str_to_encoded": CATEGORICAL_STRING_TO_ENCODED,
        "categorical_int_to_embedded": CATEGORICAL_INTEGER_TO_EMBEDDED,
        "categorical_str_to_embedded": CATEGORICAL_STRING_TO_EMBEDDED,
        "text_to_embedded": TEXT_TO_EMBEDDED,
    },
    "y_type": {
        "regression": REGRESSION_TARGETS,
        "binary": BINARY_TARGETS,
        "multiclass": MULTICLASS_TARGET_W_CLASSES,
        "multilabel": MULTILABEL_TARGET_W_LABELS,
    },
}


# %% Create inputs and encoded features ----------------------------------------
(
    inputs,
    encoded_features,
    embedded_features,
    flattened_features,
) = create_model_inputs_and_features(
    set_of_features,
    ds_train,
    print_result=True,
)
features_space = {
    "encoded": encoded_features,
    "embedded": embedded_features,
    "flattened": flattened_features,
}


# %% Base model ----------------------------------------------------------------
def model_mlp(
    inputs,
    features_space,
    optimizer,
    loss,
    metrics,
    print_summary=True,
):
    ALPHA = 0.05
    INITIALIZER = tf.keras.initializers.GlorotNormal(seed=42)
    # https://keras.io/api/layers/initializers/#henormal-class

    # You needs 'input' and 'encoded_features' as a lists
    inputs_list = list(inputs.values())
    encoded_features = list(features_space["encoded"].values())
    flattened_features = list(features_space["flattened"].values())

    # # Hot-fix for a TypeError: Tensors in list passed to 'values' of 'ConcatV2' Op
    # # have types [float32, int64] that don't all match.
    # flattened_features["description"] = tf.cast(
    #     flattened_features["description"],
    #     tf.float32,
    # )

    # * Combine different features into separated tensors
    concat_flatted = tf.keras.layers.concatenate(flattened_features, name="concat_flatted")
    btchNorm_flatted = tf.keras.layers.BatchNormalization(name="btchNorm_flatted")(concat_flatted)
    dense_flatted = tf.keras.layers.Dense(
        units=512,
        activation=tf.keras.layers.LeakyReLU(alpha=ALPHA),
        name="dense_flatted",
        kernel_initializer=INITIALIZER,
    )(btchNorm_flatted)

    concat_encoded = tf.keras.layers.concatenate(encoded_features, name="concat_encoded")
    btchNorm_encoded = tf.keras.layers.BatchNormalization(name="btchNorm_encoded")(concat_encoded)
    dense_encoded = tf.keras.layers.Dense(
        units=64,
        activation=tf.keras.layers.LeakyReLU(alpha=ALPHA),
        name="dense_encoded",
        kernel_initializer=INITIALIZER,
    )(btchNorm_encoded)

    # * Combine input features into a single tensor
    all = tf.keras.layers.concatenate([dense_encoded, dense_flatted], name="all")
    batch_0 = tf.keras.layers.BatchNormalization(name="btchNorm_0")(all)
    drop_0 = tf.keras.layers.Dropout(0.5, seed=RND_STATE, name="drop_0x50")(batch_0)

    dense_1 = tf.keras.layers.Dense(
        units=256,
        activation=tf.keras.layers.LeakyReLU(alpha=ALPHA),
        name="dense_1",
        kernel_initializer=INITIALIZER,
    )(drop_0)
    batch_1 = tf.keras.layers.BatchNormalization(name="btchNorm_1")(dense_1)
    drop_1 = tf.keras.layers.Dropout(0.5, seed=RND_STATE, name="drop_1x50")(batch_1)

    dense_2 = tf.keras.layers.Dense(
        units=128,
        activation=tf.keras.layers.LeakyReLU(alpha=ALPHA),
        name="dense_2",
        kernel_initializer=INITIALIZER,
    )(drop_1)
    batch_2 = tf.keras.layers.BatchNormalization(name="btchNorm_2")(dense_2)

    # * Output layer
    out = tf.keras.layers.Dense(
        units=1,
        activation=tf.keras.layers.LeakyReLU(alpha=ALPHA),
        name="prediction",
        kernel_initializer=INITIALIZER,
    )(batch_2)

    outputs = [out]
    model = tf.keras.Model(
        inputs_list,
        outputs,
        name="model_mlp_desc",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary(line_length=150))

    model.reset_states()
    return model


def scheduler_exp(epoch, lr):
    import math

    n_epoch = 10
    if epoch < n_epoch:
        return lr
    else:
        return lr * math.exp(-0.1)


def scheduler_drop(epoch, lr):
    n_epoch = 10
    lr_drop_rate = 0.8
    if epoch < n_epoch:
        return lr
    else:
        return lr * lr_drop_rate


# %% Build the model -----------------------------------------------------------
# Create folder for logging
log_dir = PROJECT_TRAKING_DIR / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if not Path(log_dir).is_dir():
    Path(log_dir).mkdir()

# Build & compile model
model = model_mlp(
    inputs,
    features_space,
    optimizer=MODEL_PARAMS["_optimizer"],
    loss=MODEL_PARAMS["_loss"],
    metrics=MODEL_PARAMS["_metrics"],
    print_summary=False,
)
# Plot the model
fig_model = tf.keras.utils.plot_model(
    model,
    to_file=log_dir / f"dl_{model.name}_plot.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    expand_nested=False,
    dpi=100,
    rankdir="TB",
    show_layer_activations=True,
)
display(fig_model)


# %% Get & Save model's code ---------------------------------------------------
print(f"Starting logging into {log_dir}")
with open(log_dir / f"dl_{model.name}_code.txt", "w") as f:
    print(inspect.getsource(model_mlp), file=f)

with open(log_dir / f"dl_{model.name}_params.txt", "w") as f:
    print(MODEL_PARAMS, file=f)
    print(model.optimizer.lr, file=f)

with open(log_dir / f"dl_{model.name}_features_params.json", "w") as f:
    json.dump(set_of_features, f)


# %% Train the model -----------------------------------------------------------
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_mape",
        patience=4,
        verbose=1,
        mode="min",
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=log_dir / "model",
        monitor="val_mape",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(log_dir / "logs.csv", append=False),
    tf.keras.callbacks.TerminateOnNaN(),
    # tf.keras.callbacks.LearningRateScheduler(scheduler_exp, verbose=1),
]
tf.keras.backend.clear_session()

# Train the model
history = model.fit(
    ds_train,
    batch_size=MODEL_PARAMS["_batch_size"],
    epochs=MODEL_PARAMS["_epochs"],
    verbose=1,
    validation_data=ds_val,
    shuffle=False,
    use_multiprocessing=True,
    callbacks=callbacks,
)

# Load a best model
model = tf.keras.models.load_model(log_dir / "model")

# %% Make prediction & define true values --------------------------------------
print(f"\nMake predictions for performance calculating ...")
y_pred_train = np.exp(model.predict(ds_train))
y_pred_val = np.exp(model.predict(ds_val))
y_pred_test = np.exp(model.predict(ds_test))

# ! DO NOT USE NON-TRANSFORMED TARGET !
# y_pred_train = model.predict(ds_train)
# y_pred_val = model.predict(ds_val)
# y_pred_test = model.predict(ds_test)

print(f"\nCalculate metrics for performance evaluating ...")
mae = tf.keras.losses.MeanAbsoluteError()
mape = tf.keras.losses.MeanAbsolutePercentageError()

# Train set
y_train_mae = mae(Y_true_train, y_pred_train).numpy()
y_train_mape = mape(Y_true_train, y_pred_train).numpy()
y_train_diff = 100 * abs((Y_true_train.sum() - y_pred_train.sum()) / Y_true_train.sum())
print(f"TRAIN set:")
print(f"\tMAE = {y_train_mae}\n\tMAPE = {y_train_mape}")
print(f"\t- Total Sum:")
print(f"\ttrue = {Y_true_train.sum():,}\n\tpred = {y_pred_train.sum():,}")
print(f"\t% diff = {y_train_diff}")

# Validation set
y_val_mae = mae(Y_true_val, y_pred_val).numpy()
y_val_mape = mape(Y_true_val, y_pred_val).numpy()
y_val_diff = 100 * abs((Y_true_val.sum() - y_pred_val.sum()) / Y_true_val.sum())
print(f"VAL set:")
print(f"\tMAE = {y_val_mae}\n\tMAPE = {y_val_mape}")
print(f"\t- Total Sum:")
print(f"\ttrue = {Y_true_val.sum():,}\n\tpred = {y_pred_val.sum():,}")
print(f"\t% diff = {y_val_diff}")

# Test set
y_test_mae = mae(Y_true_test, y_pred_test).numpy()
y_test_mape = mape(Y_true_test, y_pred_test).numpy()
y_test_diff = 100 * abs((Y_true_test.sum() - y_pred_test.sum()) / Y_true_test.sum())
print(f"TEST set:")
print(f"\tMAE = {y_test_mae}\n\tMAPE = {y_test_mape}")
print(f"\t- Total Sum:")
print(f"\ttrue = {Y_true_test.sum():,}\n\tpred = {y_pred_test.sum():,}")
print(f"\t% diff = {y_test_diff}")

with open(log_dir / f"dl_{model.name}_performance.txt", "w") as f:
    print(
        f"Train set: {FILE_WITH_TRAIN_DATA}\nTest set: {FILE_WITH_TEST_DATA}\n",
        file=f,
    )
    print(
        f"TRAIN set:\n\tMAE = {y_train_mae}\n\tMAPE = {y_train_mape}\n\t%DIFF = {y_train_diff}",
        file=f,
    )
    print(
        f"VAL set:\n\tMAE = {y_val_mae}\n\tMAPE = {y_val_mape}\n\t%DIFF = {y_val_diff}",
        file=f,
    )
    print(
        f"TEST set:\n\tMAE = {y_test_mae}\n\tMAPE = {y_test_mape}\n\t%DIFF = {y_test_diff}",
        file=f,
    )

print("Done.")


# %% Visualize history & prediction errors -------------------------------------
N = 10
y_true = Y_true_test
y_pred = y_pred_test

# # If you use a Modelcheckpoint Callback - this should be commented
# fig = plot_history(model.history, model.name)
# display(fig)

fig = plot_pediction_errors(y_true, y_pred, model.name, subsample=20_000)
display(fig)
fig.savefig(log_dir / "preds-errors.png")

zeros = np.count_nonzero(y_pred == 1)
print(f"Zeros predicted: # {zeros:,} - % {zeros / len(y_pred) * 100}")
print(f"Predicted minimum: {y_pred.min()}")
print(f"Predicted maximum: {y_pred.max():,}")
# q0 = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9995]
# q1 = np.quantile(y_pred, q0)
# qq = pd.DataFrame({"QQ": [str(100 * x) + "%" for x in q0], "y_pred": q1})
# display(qq)

print(f"Top {N} predictions by predicted value:")
top_n = np.argpartition(y_pred.ravel(), -N)[-N:]
df_top_n = (
    df_test.rename(columns={"value": "value_log"})
    .iloc[top_n, :]
    .assign(value_true=Y_true_test[top_n])
    .assign(value_pred=y_pred_test[top_n])
    .assign(value_diff_prc=lambda x: (x.value_true - x.value_pred) / x.value_true * 100)
)
display(df_top_n)

print(f"Bottom {N} predictions by predicted value:")
bot_n = np.argpartition(y_pred.ravel(), N)[:N]
df_bot_n = (
    df_test.rename(columns={"value": "value_log"})
    .iloc[bot_n, :]
    .assign(value_true=Y_true_test[bot_n])
    .assign(value_pred=y_pred_test[bot_n])
    .assign(value_diff_prc=lambda x: (x.value_true - x.value_pred) / x.value_true * 100)
)
display(df_bot_n)

# %% Get the errors distribution -----------------------------------------------
# def q50(x):
#     return x.quantile(0.5)


# def q90(x):
#     return x.quantile(0.9)


# df = pd.DataFrame(
#     {
#         "y_true": np.squeeze(Y_true_test),
#         "y_pred": np.squeeze(y_pred_test),
#         "weight": df_test["weight"].values,
#         "hscode_04": df_test["hscode_04"].values,
#     },
# )
# df["abs_error"] = abs(df["y_true"] - df["y_pred"])
# df["ape"] = df["abs_error"] / df["y_true"] * 100
# df["bin"] = pd.cut(df["weight"], BIN_BOUNDARIES)
# dfb_err = df.groupby(["bin"], as_index=False, observed=True).agg(
#     {
#         "ape": ["size", "mean", "min", "max", q50, q90],
#         "y_pred": ["mean", "min", "max", q50, q90],
#         "y_true": ["mean", "min", "max", q50, q90],
#         "weight": ["min", "max"],
#     }
# )
# display(dfb_err.T)

# dfc_err = df.groupby(["hscode_04"], as_index=False, observed=True).agg(
#     {
#         "ape": ["size", "mean", "min", "max", q50, q90],
#         "y_pred": ["mean", "min", "max", q50, q90],
#         "y_true": ["mean", "min", "max", q50, q90],
#     }
# )
# display(dfc_err)


# %% Save whole model ----------------------------------------------------------
model_dir = log_dir / "model"
if not Path(model_dir).is_dir():
    Path(model_dir).mkdir()
model.save(model_dir, overwrite=True, save_format=None)
# tf.keras.saving.save_model(model, model_dir, overwrite=True, save_format=None)

# %% RUN
