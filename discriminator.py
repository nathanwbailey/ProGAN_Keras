from tensorflow import keras

from model_building_blocks import MinibatchStdev, WeightedSum, wasserstein_loss


def add_discriminator_block(
    old_model: keras.models.Model, n_input_layers: int = 3
) -> list[keras.models.Model]:
    """Add a Discriminator Block to the Overall Discriminator Model."""
    init = keras.initializers.RandomNormal(stddev=0.02)
    # Each filter should have a max norm of 1.0
    const = keras.constraints.MaxNorm(1.0)
    # Get the input shape of the current model.
    in_shape = list(old_model.input.shape)

    # new input shape should be double the spatial size of the current input size
    input_shape = (in_shape[-2] * 2, in_shape[-2] * 2, in_shape[-1])
    in_image = keras.layers.Input(shape=input_shape)
    # Each Discriminator Block Consist of:
    # 2 Conv 3x3
    # Downsample layer

    # Resize to HxWx128
    d = keras.layers.Conv2D(
        128,
        1,
        padding="same",
        kernel_initializer=init,
        kernel_constraint=const,
    )(in_image)
    d = keras.layers.LeakyReLU(negative_slope=0.2)(d)

    d = keras.layers.Conv2D(
        128,
        3,
        padding="same",
        kernel_initializer=init,
        kernel_constraint=const,
    )(d)
    d = keras.layers.LeakyReLU(negative_slope=0.2)(d)
    d = keras.layers.Conv2D(
        128,
        3,
        padding="same",
        kernel_initializer=init,
        kernel_constraint=const,
    )(d)
    d = keras.layers.LeakyReLU(negative_slope=0.2)(d)

    # Create a Downsample Layer
    d = keras.layers.AveragePooling2D(pool_size=2)(d)
    # Output of the new block.
    block_new = d

    # Skip the Input layers (The block we are replacing)
    # Pass the output through the rest of the model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)

    # Create the new model with the new block
    model1 = keras.models.Model(in_image, d)

    model1.compile(
        loss=wasserstein_loss,
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8
        ),
    )

    # Add the old block to blend in
    # Downsample the input
    downsample = keras.layers.AveragePooling2D(pool_size=2)(in_image)

    # Pass through the input layers
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)

    # Blend in the new block
    d = WeightedSum()([block_old, block_new])

    # Pass the weighted sum through the rest of the network
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)

    model2 = keras.models.Model(in_image, d)
    model2.compile(
        loss=wasserstein_loss,
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8
        ),
    )
    # Return blended and straight-through model
    return [model1, model2]


def define_discriminator(
    n_blocks: int, input_shape: tuple[int, int, int] = (4, 4, 3)
) -> list[list[keras.models.Model]]:
    """Function to Create the Discriminator."""
    init = keras.initializers.RandomNormal(stddev=0.02)
    # Each filter should have a max norm of 1.0
    const = keras.constraints.MaxNorm(1.0)
    model_list: list[list[keras.models.Model]] = []
    in_image = keras.layers.Input(shape=input_shape)

    # First Block Consists of:
    # 1x1 Conv2D Layer
    # 3x3 Conv2D Layer
    # 4x4 Conv2D Layer

    d = keras.layers.Conv2D(
        128,
        1,
        padding="same",
        kernel_initializer=init,
        kernel_constraint=const,
    )(in_image)
    d = keras.layers.LeakyReLU(negative_slope=0.2)(d)
    d = MinibatchStdev()(d)

    d = keras.layers.Conv2D(
        128,
        3,
        padding="same",
        kernel_initializer=init,
        kernel_constraint=const,
    )(in_image)
    d = keras.layers.LeakyReLU(negative_slope=0.2)(d)

    d = keras.layers.Conv2D(
        128,
        4,
        padding="same",
        kernel_initializer=init,
        kernel_constraint=const,
    )(in_image)
    d = keras.layers.LeakyReLU(negative_slope=0.2)(d)

    # Dense Layer to classify
    d = keras.layers.Flatten()(d)
    out_class = keras.layers.Dense(1)(d)

    # Create the initial model
    model = keras.models.Model(in_image, out_class)

    model.compile(
        loss=wasserstein_loss,
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8
        ),
    )

    # Append to the model list
    model_list.append([model, model])

    # For n_blocks, add a discriminator block and append
    for i in range(1, n_blocks):
        old_model = model_list[i - 1][0]
        models = add_discriminator_block(old_model)
        model_list.append(models)
    return model_list
