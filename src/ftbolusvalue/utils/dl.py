""" Contains the functions used for
    features encoding \ embedding and creating DL models' inputs.

    @author: mikhail.galkin
"""

# %% Import needed python libraryies and project config info
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import PredictionErrorDisplay
from IPython.display import display


#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ U T I L S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def df_split_data(df, rnd_state, **kwargss):
    print(f"\nSplit data on train & test sets ...")
    df_train_val, df_test = model_selection.train_test_split(
        df,
        train_size=0.8,
        random_state=rnd_state,
        shuffle=True,
        **kwargss,
    )

    df_train, df_val = model_selection.train_test_split(
        df_train_val,
        train_size=0.85,
        random_state=rnd_state,
        shuffle=True,
        **kwargss,
    )

    print(f"Dev set: {df.shape[1]} vars & {df.shape[0]:,} rows.")
    print(f"\tdf_train: {df_train.shape[1]} vars & {df_train.shape[0]:,} rows.")
    print(f"\tdf_val: {df_val.shape[1]} vars & {df_val.shape[0]:,} rows.")
    print(f"\tdf_test: {df_test.shape[1]} vars & {df_test.shape[0]:,} rows.")
    return df_train, df_val, df_test


def df_to_ds(df, X_cols, Y_cols, batch_size=None, shuffle=False):
    """_summary_

    Args:
        df (Pandas DataFrame): _description_
        X_cols (list): _description_
        Y_cols (list): _description_
        batch_size (int, optional): _description_. Defaults to None.
        shuffle (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    print(f"Converting Pandas DataFrame to TF Dataset ...")
    df_X = df[X_cols].copy()

    # To use the tf.data.Dataset.from_tensor_slices()
    # we should to have tuple of np.arrayes
    # due this we convert pd.series into 2d array and put them into list
    labels = []
    for y in Y_cols:
        labels.append(np.array(df[y].values.tolist()))

    # The given tensors are sliced along their first dimension.
    # This operation preserves the structure of the input tensors,
    # removing the first dimension of each tensor and using it as the dataset dims.
    # All input tensors must have the same size in their first dimensions.
    ds_X = tf.data.Dataset.from_tensor_slices(dict(df_X))
    ds_Y = tf.data.Dataset.from_tensor_slices(tuple(labels))
    ds = tf.data.Dataset.zip((ds_X, ds_Y))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df_X), seed=42)

    # Combines consecutive elements of this dataset into batches.
    if batch_size is not None:
        ds = ds.batch(batch_size)
    return ds, labels


def scheduler_exp(epoch, lr):
    """
    Learning rate scheduler:
    This function keeps the initial learning rate for the n ten epochs
    and decreases it exponentially after that.

    Returns:
        learning rate value
    """
    n_epoch = 4
    if epoch < n_epoch:
        return lr
    else:
        return lr * math.exp(-0.1)


def scheduler_drop(epoch, lr):
    """
    Learning rate scheduler:
    This function keeps the initial learning rate for the first n epochs
    and decreases it by dropping after that.

    Returns:
        learning rate value
    """
    n_epoch = 4
    lr_drop_rate = 0.5
    if epoch < n_epoch:
        return lr
    else:
        return lr * lr_drop_rate


def eval_model(model, dataset, dataset_name):
    print(f"\nModel evaluation for: {dataset_name} ...")
    evals = model.evaluate(dataset, verbose=1)
    metrics_names = [name + f"_{dataset_name}" for name in model.metrics_names]
    eval_scores = dict(zip(metrics_names, evals))
    return eval_scores


def predict_model(model, dataset, dataset_name):
    print("\nModel prediction for: " + dataset_name)
    prediction = model.predict(dataset)
    print(dataset_name + " predicted")
    return prediction


#! -----------------------------------------------------------------------------
#! ------------------------- E N C O D I N G  ----------------------------------
#! -----------------------------------------------------------------------------
""" Function for demonstrate several types of feature column -------------------
    TensorFlow provides many types of feature columns.
    In this section, we will create several types of feature columns,
    and demonstrate how they transform a column from the dataframe.
    We will use this batch to demonstrate several types of feature columns
        example_batch = next(iter(ds_val))[0]
    A utility method to create a feature column
    and to transform a batch of data
    def demo(feature_column):
        feature_layer = keras.layers.DenseFeatures(feature_column)
        print(feature_layer(example_batch).numpy())
"""


def numeric_to_normalized(feature_name, dataset, **kwargs):
    """Int\Float values to be preprocessed via featurewise standardization
    A preprocessing which normalizes continuous features.
    https://keras.io/api/layers/preprocessing_layers/numerical/normalization/
    """
    print(f"\tNumeric: Start to Normalize ...")

    # Create model input for feature
    input = tf.keras.layers.Input(shape=(1,), name=feature_name)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, _: x[feature_name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Create a Normalization layer for our feature
    normalizer = tf.keras.layers.Normalization(
        name=f"norm-{feature_name}",
        **kwargs,
    )

    print(f"\tLearn statistics for the < {feature_name} > ...")
    normalizer.adapt(feature_ds)
    print(f"\tNormalize the input feature ...")
    encoded_feature = normalizer(input)

    #! It is possible to not use the Normalization:
    # encoded_feature = input
    print(f"\tDone: < {feature_name} >.")
    return input, encoded_feature


def numeric_to_discretized(feature_name, dataset, **kwargs):
    """Int\Float values to be discretized. By default, the discrete.
    A preprocessing which buckets continuous features by ranges.
    https://keras.io/api/layers/preprocessing_layers/numerical/discretization/
    """
    mode = kwargs["output_mode"]
    print(f"\tNumeric: Discretize & Encoding to '{mode.upper()}'...")

    # Create model input for feature
    input = tf.keras.layers.Input(shape=(1,), name=feature_name)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, _: x[feature_name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Create a Discretization layer for our feature
    discretizer = tf.keras.layers.Discretization(
        name=f"{mode}-{feature_name}",
        **kwargs,
    )

    if kwargs["num_bins"] is not None:
        print(f"\tLearn statistics for the < {feature_name} > ...")
        discretizer.adapt(feature_ds)

    print(f"\tDiscretize the input feature ...")
    encoded_feature = discretizer(input)

    if mode == "int":
        encoded_feature = tf.cast(encoded_feature, tf.float32)

    #! It is possible to not use the Discretization:
    # encoded_feature = input
    print(f"\tDone: < {feature_name} >.")
    return input, encoded_feature


def categorical_to_encoded(feature_name, dataset, is_string, **kwargs):
    """_summary_
    https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup/#stringlookup-class
    https://keras.io/api/layers/preprocessing_layers/categorical/integer_lookup/#integerlookup-class
    """
    mode = kwargs["output_mode"]
    vocabulary = kwargs["vocabulary"]

    if is_string:
        lookup_class = tf.keras.layers.StringLookup
        dtype_class = "string"
    else:
        lookup_class = tf.keras.layers.IntegerLookup
        dtype_class = "int64"
    print(f"\tCategorical {dtype_class.upper()}: Encoding to '{mode.upper()}' ...")

    # Create model input for feature
    input = tf.keras.layers.Input(
        shape=(1,),
        name=feature_name,
        dtype=dtype_class,
    )

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, _: x[feature_name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Create a Lookup layer for our feature
    lookup = lookup_class(
        name=f"{mode}-{feature_name}",
        **kwargs,
    )

    if vocabulary is None:
        print(f"\tLearn a vocabulary for the < {feature_name} > ...")
        lookup.adapt(feature_ds)
    vocab_size = lookup.vocabulary_size()
    print(f"\tGot the vocabulary with size # {vocab_size} ...")
    print(f"\tTurn the categorical input into integer indices ...")
    encoded_feature = lookup(input)

    print(f"\tDone: < {feature_name} >.")
    return input, encoded_feature


def categorical_to_embedded(feature_name, dataset, is_string, **kwargs):
    use_embedding = kwargs.pop("use_embedding")
    embed_params_ls = ["output_dim", "mask_zero", "input_length"]
    embed_params = {k: kwargs.pop(k) for k in embed_params_ls}

    mode = kwargs["output_mode"]
    if is_string:
        lookup_class = tf.keras.layers.StringLookup
        dtype_class = "string"
    else:
        lookup_class = tf.keras.layers.IntegerLookup
        dtype_class = "int64"
    print(f"\tCategorical {dtype_class.upper()}: Encoding to '{mode.upper()}' ...")

    # Create model input for feature
    input = tf.keras.layers.Input(
        shape=(1,),
        name=feature_name,
        dtype=dtype_class,
    )

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, _: x[feature_name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Create a Lookup layer for our feature.
    # Since we are not using a mask token we set mask_token to None,
    # and possible expecting any out of vocabulary (oov) token - set num_oov_indices to 1.
    # https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup/
    lookup = lookup_class(
        name=f"{mode}-{feature_name}",
        **kwargs,
    )

    vocabulary = kwargs["vocabulary"]
    if vocabulary is None:
        print(f"\tLearn a vocabulary for the < {feature_name} > ...")
        lookup.adapt(feature_ds)
    vocab_size = lookup.vocabulary_size()
    print(f"\tGot the vocabulary with size # {vocab_size:,} ...")
    print(f"\tTurn the categorical input into integer indices ...")
    encoded_feature = lookup(input)

    #! It is possible to not use the embedding.
    if use_embedding:
        print(f"\tCreate an embeddings with", end=" ")
        output_dim = embed_params.pop("output_dim")
        if output_dim is None:
            output_dim = int(math.sqrt(vocab_size))
            print(f"a calculated dimension # {output_dim} ...")
        else:
            print(f"a specified dimension # {output_dim} ...")

        embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=output_dim,
            name=f"embed-{feature_name}",
            **embed_params,
        )
        embedded_feature = embedding(encoded_feature)
        flatten = tf.keras.layers.Flatten(name=f"flat-{feature_name}")
        flattened_feature = flatten(embedded_feature)
    else:
        print(f"\tParameter 'use_embedding' = {use_embedding} ...")
        embedded_feature = encoded_feature
        flattened_feature = encoded_feature

    print(f"\tDone: < {feature_name} >.")
    return input, embedded_feature, flattened_feature


def text_to_embedded(feature_name, dataset, **kwargs):
    use_embedding = kwargs.pop("use_embedding")
    flat_or_pool = kwargs.pop("flat_or_pool")
    embed_params_ls = ["output_dim", "mask_zero", "input_length"]
    embed_params = {k: kwargs.pop(k) for k in embed_params_ls}

    mode = kwargs["output_mode"]
    print(f"\tText: Vectorization with '{mode.upper()}' mode ...")

    # Create model input for feature
    input = tf.keras.layers.Input(
        shape=(1,),
        name=feature_name,
        dtype="string",
    )

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, _: x[feature_name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Create a TextVectorization layer for our feature
    # https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/
    text_vectorizer = tf.keras.layers.TextVectorization(
        name=f"textvect-{feature_name}",
        **kwargs,
    )

    print(f"\tLearn a text for the < {feature_name} > ...")
    text_vectorizer.adapt(feature_ds)

    vocab_size = text_vectorizer.vocabulary_size()
    print(f"\tGot the text vocabulary with size # {vocab_size:,} ...")
    print(f"\tTurn the text input into integer indices ...")
    encoded_feature = text_vectorizer(input)

    #! It is possible to not use the embedding.
    if use_embedding:
        print(f"\tCreate an embeddings with", end=" ")
        output_dim = embed_params.pop("output_dim")
        if output_dim is None:
            output_dim = int(math.sqrt(vocab_size))
            print(f"a calculated dimension # {output_dim} ...")
        else:
            print(f"a specified dimension # {output_dim} ...")

        embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=output_dim,
            name=f"embed-{feature_name}",
        )
        embedded_feature = embedding(encoded_feature)

        if flat_or_pool == "avgpool":
            # To fix "ResourceExhaustedError: OOM when allocating tensor"
            # and continue to use GPU
            flatten = tf.keras.layers.GlobalAveragePooling1D(
                data_format="channels_first",
                name=f"avgpool-{feature_name}",
            )
        elif flat_or_pool == "maxpool":
            # To fix "ResourceExhaustedError: OOM when allocating tensor"
            # and continue to use GPU
            flatten = tf.keras.layers.GlobalMaxPooling1D(
                data_format="channels_first",
                name=f"maxpool-{feature_name}",
            )
        else:
            flatten = tf.keras.layers.Flatten(name=f"flat-{feature_name}")

        flattened_feature = flatten(embedded_feature)
    else:
        print(f"\tParameter 'use_embedding' = {use_embedding} ...")
        embedded_feature = encoded_feature
        flattened_feature = encoded_feature

    print(f"\tDone: < {feature_name} >.")
    return input, embedded_feature, flattened_feature


def create_model_inputs_and_features(
    set_of_features,
    dataset,
    print_result=True,
):
    print(f"\nCreate Inputs + Encoded & Embedded & Flattened features ...")
    inputs = {}
    encoded_features = {}
    embedded_features = {}
    flattened_features = {}

    x_type = set_of_features["x_type"]
    for feature_name in set_of_features["x"]:
        print(f"Feature: < {feature_name} > :")

        if feature_name in x_type["numeric_to_normalized"]:
            params = x_type["numeric_to_normalized"][feature_name]
            input, encoded_feature = numeric_to_normalized(
                feature_name,
                dataset,
                **params,
            )
            encoded_features[feature_name] = encoded_feature

        if feature_name in x_type["numeric_to_discretized"]:
            params = x_type["numeric_to_discretized"][feature_name]
            input, encoded_feature = numeric_to_discretized(
                feature_name,
                dataset,
                **params,
            )
            encoded_features[feature_name] = encoded_feature

        if feature_name in x_type["categorical_int_to_encoded"]:
            params = x_type["categorical_int_to_encoded"][feature_name]
            input, encoded_feature = categorical_to_encoded(
                feature_name,
                dataset,
                is_string=False,
                **params,
            )
            encoded_features[feature_name] = encoded_feature

        if feature_name in x_type["categorical_str_to_encoded"]:
            params = x_type["categorical_str_to_encoded"][feature_name]
            input, encoded_feature = categorical_to_encoded(
                feature_name,
                dataset,
                is_string=True,
                **params,
            )
            encoded_features[feature_name] = encoded_feature

        if feature_name in x_type["categorical_int_to_embedded"]:
            params = x_type["categorical_int_to_embedded"][feature_name]
            input, embedded_feature, flattened_feature = categorical_to_embedded(
                feature_name,
                dataset,
                is_string=False,
                **params,
            )
            embedded_features[feature_name] = embedded_feature
            flattened_features[feature_name] = flattened_feature

        if feature_name in x_type["categorical_str_to_embedded"]:
            params = x_type["categorical_str_to_embedded"][feature_name]
            input, embedded_feature, flattened_feature = categorical_to_embedded(
                feature_name,
                dataset,
                is_string=True,
                **params,
            )
            embedded_features[feature_name] = embedded_feature
            flattened_features[feature_name] = flattened_feature

        if feature_name in x_type["text_to_embedded"]:
            params = x_type["text_to_embedded"][feature_name]
            input, embedded_feature, flattened_feature = text_to_embedded(
                feature_name,
                dataset,
                **params,
            )
            embedded_features[feature_name] = embedded_feature
            flattened_features[feature_name] = flattened_feature

        inputs[feature_name] = input

    if print_result:
        print(f"\nPrepared Inputs:")
        display(inputs)
        print(f"Encoded features:")
        display(encoded_features)
        print(f"Embedded features:")
        display(embedded_features)
        print(f"Flattened features:")
        display(flattened_features)
    print(f"Model's inputs and feature encoding have been done.")
    return inputs, encoded_features, embedded_features, flattened_features


#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~ P L O T S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def plot_history_loss(history, nn_name, loss="loss"):
    plt.figure(figsize=(12, 8))
    # * Var#1: Use a log scale to show the wide range of values.
    # plt.semilogy(
    #     history.epoch, history.history[loss], label="Train: " + nn_name
    # )
    # plt.semilogy(
    #     history.epoch,
    #     history.history[f"val_{loss}"],
    #     label="Val: " + nn_name,
    #     linestyle="--",
    # )
    # * Var#2: W/o a log scaling.
    plt.plot(history.epoch, history.history[loss], label="Train: " + nn_name)
    plt.plot(
        history.epoch,
        history.history[f"val_{loss}"],
        label="Val: " + nn_name,
        linestyle="--",
    )

    plt.title(nn_name + f": model {loss}")
    plt.xlabel("epoch")
    plt.ylabel(loss)
    plt.legend()
    # get current figure
    fig = plt.gcf()
    plt.close()
    return fig


def plot_history(history, nn_name):
    metrics = [x for x in history.history.keys() if x[:3] != "val"]
    nrows = math.ceil(len(metrics) / 2)
    ncols = 2

    plt.figure(figsize=(14, 7 * nrows))
    for n, metric in enumerate(metrics):
        print
        plt.subplot(nrows, ncols, n + 1)
        plt.plot(
            history.epoch,
            history.history[metric],
            label="Train set",
        )
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            linestyle="--",
            label="Val set",
        )
        plt.xlabel("epoch")
        plt.ylabel(metric)
        y_min = 0.99 * min(
            min(history.history[metric]),
            min(history.history["val_" + metric]),
        )
        y_max = 1.01 * max(
            max(history.history[metric]),
            max(history.history["val_" + metric]),
        )
        plt.ylim([y_min, y_max])
        plt.legend()

    # get current figure
    fig = plt.gcf()

    # create main title
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.suptitle(f"{nn_name} : Visualize model history")

    plt.close()
    return fig


def plot_pediction_errors(y_true, y_pred, nn_name, subsample=14_000):
    fig, axs = plt.subplots(ncols=2, figsize=(14, 7))
    PredictionErrorDisplay.from_predictions(
        y_true,
        y_pred,
        kind="actual_vs_predicted",
        subsample=subsample,
        ax=axs[0],
        random_state=42,
    )
    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(
        y_true,
        y_pred,
        kind="residual_vs_predicted",
        subsample=subsample,
        ax=axs[1],
        random_state=42,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    fig.suptitle(f"{nn_name} : Visualize prediction errors")
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    return fig
