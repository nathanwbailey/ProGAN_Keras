from tensorflow import keras

from model_building_blocks import wasserstein_loss


def define_composite_model(
    discriminators: list[list[keras.models.Model]],
    generators: list[list[keras.models.Model]],
) -> list[keras.models.Model]:
    """Function to create the Overall ProGAN Model."""
    model_list: list[list[keras.models.Model]] = []
    for i in range(  # pylint: disable=consider-using-enumerate
        len(discriminators)
    ):
        # Loop through the generators/discriminators and create the 2 models.
        # 0 indexed model is the straight through model
        # 1 indexed model is the blended model
        g_models, d_models = generators[i], discriminators[i]
        d_models[0].trainable = False
        model1 = keras.Sequential()
        model1.add(g_models[0])
        model1.add(d_models[0])
        model1.compile(
            loss=wasserstein_loss,
            optimizer=keras.optimizers.Adam(
                learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8
            ),
        )

        d_models[1].trainable = False
        model2 = keras.Sequential()
        model2.add(g_models[1])
        model2.add(d_models[1])
        model2.compile(
            loss=wasserstein_loss,
            optimizer=keras.optimizers.Adam(
                learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8
            ),
        )
        model_list.append([model1, model2])
    return model_list
