"""
    Base, a very simple Deep Learning model for a benchmark.

    @author: mikhail.galkin
"""

#### WARNING: the code that follows would make you cry.
#### A Safety Pig is provided below for your benefit
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
import inspect
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import model_selection
from pprint import pprint
from IPython.display import display


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
    "hscode_04_desc_short": "object",
    "weight": "int32",
    "value": "int32",
}
USECOLS = list(COLS_DTYPE.keys())

X_COLS = [
    "ade_month",
    "hscode_04",
    "hscode_04_desc_short",
    "weight",
]
Y_COLS = ["value"]

MODEL_PARAMS = {
    "_batch_size": 65_536,
    "_epochs": 20,
    "_optimizer": tf.keras.optimizers.RMSprop(learning_rate=0.001),
    "_loss": tf.keras.losses.MeanSquaredError(name="mse"),  # +
    # "_loss": tf.keras.losses.MeanAbsoluteError(name="mae"), # -
    # "_loss": tf.keras.losses.MeanAbsolutePercentageError(name="mape"),  # +++
    # "_loss": tf.keras.losses.MeanSquaredLogarithmicError(name="msle"), # ++
    "_metrics": tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
}

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

# print(f"Remove duplicates:")
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
df_train[Y_COLS] = np.log(df_train[Y_COLS])
df_val[Y_COLS] = np.log(df_val[Y_COLS])
df_test[Y_COLS] = np.log(df_test[Y_COLS])


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
# A lists of the binary feature names. Feature like "is_animal"
BINARY_FEATURES = []  # ["is_animal"]

# A dictionaries of the features' names and **kwargs for preprocessing layers.
NUMERIC_TO_NORMALIZED = {
    "weight": {
        "axis": -1,
        "mean": None,
        "variance": None,
    }
}
NUMERIC_TO_DISCRETIZED = {
    None: {
        "bin_boundaries": None,
        "num_bins": 64,  # max = 100
        "output_mode": "int",  # ("int", "one_hot", "multi_hot", "count")
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
        "output_mode": "int",  # ("int", "one_hot", "multi_hot", "count", "tf_idf")
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
        "output_mode": "one_hot",  # ("int", "one_hot", "multi_hot", "count", "tf_idf")
    }
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
        "output_mode": "int",  # ("int", "one_hot", "multi_hot", "count", "tf_idf")
        # * Related to Embedding layer:
        "output_dim": None,
        "mask_zero": False,
        "input_length": None,
    }
}
CATEGORICAL_STRING_TO_EMBEDDED = {
    "hscode_04": {
        "use_embedding": True,
        # * Related to Lookup layer:
        "max_tokens": None,
        "num_oov_indices": 1,
        "mask_token": None,
        "oov_token": "[UNK]",
        "vocabulary": sorted(df_train["hscode_04"].unique().tolist()),
        "output_mode": "int",  # ("int", "one_hot", "multi_hot", "count", "tf_idf")
        # * Related to Embedding layer:
        "output_dim": None,
        "mask_zero": False,
        "input_length": None,
    }
}

# A list of textual features w/o vocabulary we want be embedded
TEXT_TO_EMBEDDED = {
    "hscode_04_desc_short": {
        "use_embedding": True,
        # * Related to Lookup layer:
        "max_tokens": None,  # 20_000 good choice for big text - Not necessary
        "standardize": "lower_and_strip_punctuation",
        "split": "whitespace",  # (None, "character")
        "ngrams": None,
        "output_mode": "int",  # ("int", "multi_hot", "count", "tf_idf")
        "output_sequence_length": 16,  # Only valid in "int" mode
        "pad_to_max_tokens": False,  # Only valid in "multi_hot", "count", "tf_idf"
        "vocabulary": None,
        "idf_weights": None,  # Only valid in "tf_idf" mode
        "sparse": False,  # Bool. Only applicable to "int" mode
        "ragged": False,  # Bool. Only applicable to "multi_hot", "count", "tf_idf" modes
        # * Related to Embedding layer:
        "output_dim": None,
        "mask_zero": False,
        "input_length": None,
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
        "categorical_int_to_embedded": CATEGORICAL_INTEGER_TO_EMBEDDED,
        "categorical_str_to_encoded": CATEGORICAL_STRING_TO_ENCODED,
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
def base_model(
    inputs,
    features_space,
    optimizer,
    loss,
    metrics,
    print_summary=True,
):
    # You needs 'input' and 'encoded_features' as a lists
    inputs_list = list(inputs.values())
    encoded_features = list(features_space["encoded"].values())
    flattened_features = list(features_space["flattened"].values())
    features_list = encoded_features + flattened_features

    # Combine input features into a single tensor
    all = tf.keras.layers.concatenate(features_list, name="all")
    batch_1 = tf.keras.layers.BatchNormalization(name="btchNorm_1")(all)

    # https://keras.io/api/layers/initializers/#henormal-class
    initializer = tf.keras.initializers.GlorotNormal(seed=42)

    dense_1 = tf.keras.layers.Dense(
        units=256,
        activation="relu",
        name="dense_1",
        kernel_initializer=initializer,
    )(batch_1)
    batch_2 = tf.keras.layers.BatchNormalization(name="btchNorm_2")(dense_1)

    # Output layer
    out_value = tf.keras.layers.Dense(
        units=1,
        activation="relu",
        name="pred_value",
        kernel_initializer=initializer,
    )(batch_2)

    outputs = [out_value]
    model = tf.keras.Model(
        inputs_list,
        outputs,
        name="base_model",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary(line_length=150))

    model.reset_states()
    return model


# %% Build the model -----------------------------------------------------------
# Create folder for logging
log_dir = PROJECT_TRAKING_DIR / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if not Path(log_dir).is_dir():
    Path(log_dir).mkdir()

# Build & compile model
model = base_model(
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


# %% Get & save model's code ---------------------------------------------------
with open(log_dir / f"dl_{model.name}_code.txt", "w") as f:
    print(inspect.getsource(base_model), file=f)

with open(log_dir / f"dl_{model.name}_params.txt", "w") as f:
    print(MODEL_PARAMS, file=f)
    print(model.optimizer.lr, file=f)


# %% Train the model -----------------------------------------------------------
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
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


# %% Make prediction & define true values --------------------------------------
print(f"\nMake predictions for performance calculating ...")
y_pred_train = np.exp(model.predict(ds_train))
y_pred_val = np.exp(model.predict(ds_val))
y_pred_test = np.exp(model.predict(ds_test))

print(f"\nCalculate metrics for performance evaluating ...")
mae = tf.keras.losses.MeanAbsoluteError()
mape = tf.keras.losses.MeanAbsolutePercentageError()

# Train set
y_train_mae = mae(Y_true_train, y_pred_train).numpy()
y_train_mape = mape(Y_true_train, y_pred_train).numpy()
y_train_diff = 100 * abs((Y_true_train.sum() - y_pred_train.sum()) / Y_true_train.sum())
print(f"TRAIN set:\n\tMAE = {y_train_mae}\n\tMAPE = {y_train_mape}")
print(f"\t- Total Sum:")
print(f"\ttrue = {Y_true_train.sum():,}\n\tpred = {y_pred_train.sum():,}")
print(f"\t% diff = {y_train_diff}")

# Validation set
y_val_mae = mae(Y_true_val, y_pred_val).numpy()
y_val_mape = mape(Y_true_val, y_pred_val).numpy()
y_val_diff = 100 * abs((Y_true_val.sum() - y_pred_val.sum()) / Y_true_val.sum())
print(f"VAL set:\n\tMAE = {y_val_mae}\n\tMAPE = {y_val_mape}")
print(f"\t- Total Sum:")
print(f"\ttrue = {Y_true_val.sum():,}\n\tpred = {y_pred_val.sum():,}")
print(f"\t% diff = {y_val_diff}")

# Test set
y_test_mae = mae(Y_true_test, y_pred_test).numpy()
y_test_mape = mape(Y_true_test, y_pred_test).numpy()
y_test_diff = 100 * abs((Y_true_test.sum() - y_pred_test.sum()) / Y_true_test.sum())
print(f"TEST set:\n\tMAE = {y_test_mae}\n\tMAPE = {y_test_mape}")
print(f"\t- Total Sum:")
print(f"\ttrue = {Y_true_test.sum():,}\n\tpred = {y_pred_test.sum():,}")
print(f"\t% diff = {y_test_diff}")

with open(log_dir / f"dl_{model.name}_performance.txt", "w") as f:
    print(
        f"Train set: {FILE_WITH_TRAIN_DATA}\n\tTest set: {FILE_WITH_TEST_DATA}\n",
        file=f,
    )
    print(history.history, file=f)
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
y_true = Y_true_test
y_pred = y_pred_test

fig = plot_history(history, model.name)
display(fig)

fig = plot_pediction_errors(y_true, y_pred, model.name, subsample=20_000)
display(fig)

zeros = np.count_nonzero(y_pred == 1)
print(f"Zeros predicted: # {zeros:,} - % {zeros / len(y_pred) * 100}")
print(f"Predicted minimum: {y_pred.min()}")


# %% Baseline model ------------------------------------------------------------
print(f"-" * 80)
print(f"\nCompare the performance with baseline model ...")

# Train set
y_pred_train_bl = np.asarray([Y_true_train.mean()] * len(Y_true_train))
y_pred_train_bl = y_pred_train_bl.reshape(len(y_pred_train_bl), 1)
y_train_mae_bl = mae(Y_true_train, y_pred_train_bl).numpy()
y_train_mape_bl = mape(Y_true_train, y_pred_train_bl).numpy()
print(f"TRAIN set:\n\t MAE = {y_train_mae_bl}\n\t MAPE = {y_train_mape_bl}")

# Validation set
y_pred_val_bl = np.array([Y_true_val.mean()] * len(Y_true_val))
y_pred_val_bl = y_pred_val_bl.reshape(len(y_pred_val_bl), 1)
y_val_mae_bl = mae(Y_true_val, y_pred_val_bl).numpy()
y_val_mape_bl = mape(Y_true_val, y_pred_val_bl).numpy()
print(f"VAL set:\n\t MAE = {y_val_mae_bl}\n\t MAPE = {y_val_mape_bl}")

# Test set
y_pred_test_bl = np.array([Y_true_test.mean()] * len(Y_true_test))
y_pred_test_bl = y_pred_test_bl.reshape(len(y_pred_test_bl), 1)
y_test_mae_bl = mae(Y_true_test, y_pred_test_bl).numpy()
y_test_mape_bl = mape(Y_true_test, y_pred_test_bl).numpy()
print(f"TEST set:\n\t MAE = {y_test_mae_bl}\n\t MAPE = {y_test_mape_bl}")

# %% RUN
