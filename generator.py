from tensorflow import keras
from model_building_blocks import PixelNormalization
from model_building_blocks import WeightedSum

def add_generator_block(old_model: keras.models.Model) -> list[keras.models.Model]:
    init = keras.initializers.RandomNormal(stddev=0.02)
    # Each filter should have a max norm of 1.0
    const = keras.constraints.MaxNorm(1.0)
    block_end = old_model.layers[-2].output


    upsampling = keras.layers.UpSampling2D()(block_end)
    g = keras.layer.Conv2D(128, 3, padding='same', kernel_initializer=init, kernel_constraint=const)(upsampling)
    g = PixelNormalization(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)
    g = keras.layer.Conv2D(128, 3, padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)

    out_image = keras.layers.Conv2D(3, 1, padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    model1 = keras.models.Model(old_model.input, out_image)
    out_old = old_model.layers[-1]
    out_image2 = out_old(upsampling)
    merged = WeightedSum()([out_image2, out_image])
    model2 = keras.models.Model(old_model.input, merged)
    return [model1, model2]

def define_generator(latent_dim: int, n_blocks: int, in_dim=4) -> list[list[keras.models.Model]]:
    init = keras.initializers.RandomNormal(stddev=0.02)
    # Each filter should have a max norm of 1.0
    const = keras.constraints.MaxNorm(1.0)

    model_list = [] # type: list[list[keras.models.Model]]
    in_latent = keras.layers.Input(shape=(latent_dim,))
    g = keras.layers.Dense(128 * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const)(in_latent)
    g = keras.layers.Reshape((in_dim, in_dim, 128))(g)

    g = keras.layers.Conv2D(128, 4, padding='same',kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)

    g = keras.layers.Conv2D(128, 3, padding='same',kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)

    out_image = keras.layers.Conv2D(3, 1, padding='same',kernel_initializer=init, kernel_constraint=const)(g)
    model = keras.models.Model(in_latent, out_image)
    model_list.append([model, model])

    for i in range(1, n_blocks):
        old_model = model_list[i-1][0]
        models = add_generator_block(old_model=old_model)
        model_list.append(models)

    return model_list
