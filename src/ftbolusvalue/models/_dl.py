""" Contains the DL models' architectures
    and functions used for models' creating, training and evaluating.

    @author: mikhail.galkin
"""
# %% Import needed python libraryies and project config info
import tensorflow as tf

#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~ M O D E L S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# %% ---------------------------------------------------------------------------
def create_mlp_block(
    hidden_units,
    dropout_rate,
    activation,
    normalization_layer,
    name=None,
):
    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer),
        mlp_layers.append(tf.keras.layers.Dense(units, activation=activation))
        mlp_layers.append(tf.keras.layers.Dropout(dropout_rate))

    return tf.keras.Sequential(mlp_layers, name=name)


def transformer_model(
    inputs,
    encoded_features,
    embedded_features,
    embedding_dims,  # Size of each attention head for query and key.
    num_transformer_blocks,
    num_heads,  #  Number of attention heads.
    dropout_rate,  # Dropout probability.
    optimizer,
    loss,
    metrics,
    print_summary=True,
):
    # You needs 'input' and 'encoded_features' as a lists
    inputs_list = list(inputs.values())
    encoded_features_list = list(encoded_features.values())
    embedded_features_list = list(embedded_features.values())

    # Stack categorical feature embeddings for the Tansformer.
    embedded_features_stacked = tf.stack(
        embedded_features_list,
        axis=1,
        name="embedded_stacked",
    )

    # Create multiple layers of the Transformer block.
    for block_idx in range(num_transformer_blocks):
        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dims,
            dropout=dropout_rate,
            name=f"multihead_attention_{block_idx}",
        )(embedded_features_stacked, embedded_features_stacked)
        # Skip connection #1
        x = tf.keras.layers.Add(name=f"skip_connection1_{block_idx}")(
            [attention_output, embedded_features_stacked]
        )
        # Layer normalization #1
        x = tf.keras.layers.LayerNormalization(
            name=f"norm1_{block_idx}",
            epsilon=1e-6,
        )(x)
        # Feedforward block
        feedforward_output = create_mlp_block(
            hidden_units=[embedding_dims],
            dropout_rate=dropout_rate,
            activation=tf.keras.activations.gelu,
            normalization_layer=tf.keras.layers.LayerNormalization(epsilon=1e-6),
            name=f"mlp_block_{block_idx}",
        )(x)
        # Skip connection #2
        x = tf.keras.layers.Add(name=f"skip_connection2_{block_idx}")(
            [feedforward_output, x],
        )
        # Layer normalization #2
        embedded_features_stacked = tf.keras.layers.LayerNormalization(
            name=f"norm2_{block_idx}", epsilon=1e-6
        )(x)

    # Flatten the "contextualized" embeddings of the categorical features.
    embedded_features_flattened = tf.keras.layers.Flatten()(
        embedded_features_stacked,
        # name="flat_embedded",
    )

    # Concatenate encoded (numerical) features.
    encoded_features_concatenated = tf.keras.layers.concatenate(
        encoded_features_list,
        name="concat_encoded",
    )
    # Apply layer normalization to the numerical encoded features.
    encoded_features_normalized = tf.keras.layers.LayerNormalization(
        name="norm_encoded",
        epsilon=1e-6,
    )(encoded_features_concatenated)

    # Concatenate all features into a single tensor
    all = tf.keras.layers.concatenate(
        [embedded_features_flattened, encoded_features_normalized],
        name="all",
    )
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
    out_pred_sales = tf.keras.layers.Dense(
        units=1,
        activation="relu",
        name="pred_sales",
        kernel_initializer=initializer,
    )(batch_2)

    outputs = [out_pred_sales]
    model = tf.keras.Model(
        inputs_list,
        outputs,
        name="transformer_model",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary(line_length=150))

    model.reset_states()
    return model
