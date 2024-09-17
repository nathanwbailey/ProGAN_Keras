from tensorflow import keras

from model_building_blocks import PixelNormalization, WeightedSum


def add_generator_block(
    old_model: keras.models.Model,
) -> list[keras.models.Model]:
    """Add a Generator Block to the Overall Generator Model."""

    # A generator block consists of:
    # Upsampling Layer
    # 2 3x3 Conv2D layers

    init = keras.initializers.RandomNormal(stddev=0.02)
    # Each filter should have a max norm of 1.0
    const = keras.constraints.MaxNorm(1.0)
    # Get the output from the Upsampling layer of the current generator
    block_end = old_model.layers[-2].output

    # Add an upsampling layer to double the spatial size
    upsampling = keras.layers.UpSampling2D()(block_end)
    # Add Conv Layers
    g = keras.layers.Conv2D(
        128,
        3,
        padding="same",
        kernel_initializer=init,
        kernel_constraint=const,
    )(upsampling)
    g = PixelNormalization()(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)
    g = keras.layers.Conv2D(
        128,
        3,
        padding="same",
        kernel_initializer=init,
        kernel_constraint=const,
    )(g)
    g = PixelNormalization()(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)

    # Create a New Output layer (outputs the image)
    out_image = keras.layers.Conv2D(
        3, 1, padding="same", kernel_initializer=init, kernel_constraint=const
    )(g)

    # Create the New Model
    model1 = keras.models.Model(old_model.input, out_image)
    # Get the old output layer
    out_old = old_model.layers[-1]
    # Connect this to the output of the new upsampling layer
    out_image2 = out_old(upsampling)
    # Compute the Weighted sum of the old output and new output
    merged = WeightedSum()([out_image2, out_image])
    # Output of this forms the new generator model
    model2 = keras.models.Model(old_model.input, merged)
    # Return both the plain new model and the blended new model
    return [model1, model2]


def define_generator(
    latent_dim: int, n_blocks: int, in_dim: int = 4
) -> list[list[keras.models.Model]]:
    """Function to Construct the Full Generator."""
    init = keras.initializers.RandomNormal(stddev=0.02)
    # Each filter should have a max norm of 1.0
    const = keras.constraints.MaxNorm(1.0)

    # Model List is a List of a List of Models
    model_list: list[list[keras.models.Model]] = []
    # Input is from a Normal Distribution of size latent_dim
    in_latent = keras.layers.Input(shape=(latent_dim,))
    # Pass through a Dense Layer
    g = keras.layers.Dense(
        128 * in_dim * in_dim,
        kernel_initializer=init,
        kernel_constraint=const,
    )(in_latent)
    # Reshape
    g = keras.layers.Reshape((in_dim, in_dim, 128))(g)

    # First Generator Block Consists of:
    # 4x4 Conv
    # 3x3 Conv
    # 1x1 Conv (Output Layer)

    g = keras.layers.Conv2D(
        128,
        4,
        padding="same",
        kernel_initializer=init,
        kernel_constraint=const,
    )(g)
    g = PixelNormalization()(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)

    g = keras.layers.Conv2D(
        128,
        3,
        padding="same",
        kernel_initializer=init,
        kernel_constraint=const,
    )(g)
    g = PixelNormalization()(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)

    # Output Layer
    out_image = keras.layers.Conv2D(
        3, 1, padding="same", kernel_initializer=init, kernel_constraint=const
    )(g)
    # Create the starting Model
    model = keras.models.Model(in_latent, out_image)
    # Append the Model to the List
    model_list.append([model, model])

    # For the number of blocks generate the additional blocks
    # Append to the List
    for i in range(1, n_blocks):
        old_model = model_list[i - 1][0]
        models = add_generator_block(old_model=old_model)
        model_list.append(models)

    return model_list
