from typing import Any

import numpy as np
from numpy.typing import NDArray
from tensorflow import keras

from model_building_blocks import WeightedSum


def load_real_samples(filename: str) -> NDArray[np.float32]:
    """Load the Samples from a Zipped Dataset."""
    data = np.load(filename)
    x = data["arr_0"]
    x = x.astype("float32")
    x = (x - 127.5) / 127.5
    return x  # type: ignore[no-any-return]


def generate_real_samples(
    dataset: NDArray[Any], n_samples: int
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Generate Real Samples (label is 1)."""
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    x = dataset[ix]
    y = np.ones((n_samples, 1))
    return x, y


def generate_latent_points(latent_dim: int, n_samples: int) -> NDArray[Any]:
    """Sample from a Normal Distribution."""
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(
    generator: keras.models.Model, latent_dim: int, n_samples: int
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Generate Fake Samples (label is -1)"""
    x_input = generate_latent_points(latent_dim, n_samples)
    x = generator.predict(x_input)
    y = np.ones((n_samples, 1)) * -1.0
    return x, y


def update_fadein(
    models: list[keras.models.Model], step: int, n_steps: int
) -> None:
    """Update Alpha Parameter for all WeightedSum Layers."""
    alpha = step / float(n_steps - 1)
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                layer.alpha.assign(alpha)


def scale_dataset(
    images: NDArray[Any], new_shape: tuple[int, int]
) -> NDArray[Any]:
    """Scale the Images in the Dataset to a Specified Size."""
    image_list: list[NDArray[Any]] = []
    for image in images:
        image_list.append(np.resize(image, new_shape))
    return np.asarray(image_list)
