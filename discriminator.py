from tensorflow import keras
from model_building_blocks import wasserstein_loss
from model_building_blocks import WeightedSum
from model_building_blocks import MinibatchStdev

def add_discriminator_block(old_model: keras.models.Model, n_input_layers: int = 3) -> list[keras.models.Model]:
    init = keras.initializers.RandomNormal(stddev=0.02)
    # Each filter should have a max norm of 1.0
    const = keras.constraints.MaxNorm(1.0)
    in_shape = list(old_model.input.shape)

    # new input shape should be double the size
    input_shape = (in_shape[-2]*2, in_shape[-2]*2, in_shape[-1])
    in_image = keras.layers.Input(shape=input_shape)

    d = keras.layers.Conv2D(128, 1, padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)

    d = keras.layers.Conv2D(128, 3, padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)
    d = keras.layers.Conv2D(128, 3, padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)

    d = keras.layers.AveragePooling2D()(d)
    block_new = d

    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    model1 = keras.models.Model(in_image, d)
    model1.compile(loss=wasserstein_loss, optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    downsample = keras.layers.AveragePooling2D()(in_image)

    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    d = WeightedSum()[block_old, block_new]

    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)

    model2 = keras.layers.Model(in_image, d)
    model2.compile(loss=wasserstein_loss, optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    return [model1, model2]

def define_discriminator(n_blocks: int, input_shape=(4, 4, 3)):
    init = keras.initializers.RandomNormal(stddev=0.02)
    # Each filter should have a max norm of 1.0
    const = keras.constraints.MaxNorm(1.0)
    model_list = [] # type: list[list[keras.models.Model]]
    in_image = keras.layers.Input(shape=input_shape)

    d = keras.layers.Conv2D(128, 1, padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)
    d = MinibatchStdev()(d)

    d = keras.layers.Conv2D(128, 3, padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)

    d = keras.layers.Conv2D(128, 4, padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)

    d = keras.layers.Flatten()(d)
    out_class = keras.layers.Dense(1)(d)
    model = keras.models.Model(in_image, out_class)

    model.compile(loss=wasserstein_loss, optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    model_list.append([model, model])

    for i in range(1, n_blocks):
        old_model = model_list[i - 1][0]
        models = add_discriminator_block(old_model)
        model_list.append(models)
    return model_list