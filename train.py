from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tensorflow import keras

from helper_functions import (generate_fake_samples, generate_latent_points,
                              generate_real_samples, scale_dataset,
                              update_fadein)


def summarize_performance(
    status: str,
    g_model: keras.models.Model,
    latent_dim: int,
    n_samples: int = 25,
) -> None:
    """Function to Generate Images and Save Model."""
    gen_shape = g_model.output_shape
    name = f"{gen_shape[1]}x{gen_shape[2]}-{status}"

    x, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    x = (x - x.min()) / (x.max() - x.min())

    square = int(np.sqrt(n_samples))
    for i in range(n_samples):
        plt.subplot(square, square, 1 + i)
        plt.axis("off")
        plt.imshow(x[i])

    filename1 = f"image_plots/plot_{name}.png"
    plt.savefig(filename1)
    plt.close()

    g_model.save(f"models/model_{name}.keras")


def train_epochs(
    g_model: keras.models.Model,
    d_model: keras.models.Model,
    gan_model: keras.models.Model,
    dataset: NDArray[Any],
    n_epochs: int,
    n_batch: int,
    latent_dim: int,
    fadein: bool = False,
) -> None:
    """Train the GAN for a set Number of Epochs."""
    # n_batch = number of samples per batch
    # Get the number of batches
    batch_per_epoch = int(dataset.shape[0] / n_batch)
    # Get the total number of steps
    n_steps = batch_per_epoch * n_epochs
    half_batch = int(n_batch / 2)

    print(f"Number of Steps: {n_steps}")

    # Loop through the steps
    for i in range(n_steps):
        # Update alpha if needed
        if fadein:
            update_fadein([g_model, d_model], i, n_steps)

        # Generate a half batch of real samples
        x_real, y_real = generate_real_samples(dataset, half_batch)
        # Generate a half batch of fake samples
        x_fake, y_fake = generate_fake_samples(
            g_model, latent_dim, half_batch
        )

        # Train the discriminator on the real and fake samples
        d_model.trainable = True
        d_loss1 = d_model.train_on_batch(x_real, y_real)
        d_loss2 = d_model.train_on_batch(x_fake, y_fake)
        d_model.trainable = False

        # Generate some latent data
        z_input = generate_latent_points(latent_dim, n_batch)
        y_real2 = np.ones((n_batch, 1))

        # Train the generator
        # Pass though the GAN model with labels set to 1
        g_loss = gan_model.train_on_batch(z_input, y_real2)[0]

        print(
            f"Batch: {i+1}, d_loss1: {d_loss1:4f}, d_loss2: {d_loss2:4f}, g_loss: {g_loss:4f}"
        )


def train(
    g_models: list[list[keras.models.Model]],
    d_models: list[list[keras.models.Model]],
    gan_models: list[list[keras.models.Model]],
    dataset: NDArray[Any],
    latent_dim: int,
    e_norm: list[int],
    e_fadein: list[int],
    n_batch: list[int],
) -> None:
    """Overall Training Function."""
    # Get the initial models
    g_normal, d_normal, gan_normal = (
        g_models[0][0],
        d_models[0][0],
        gan_models[0][0],
    )
    gen_shape = g_normal.output_shape

    scaled_data = scale_dataset(dataset, gen_shape[1:])
    print(f"Scaled Data: {scaled_data.shape}")
    # Train the initial model
    train_epochs(
        g_normal,
        d_normal,
        gan_normal,
        scaled_data,
        e_norm[0],
        n_batch[0],
        latent_dim,
    )
    summarize_performance("tuned", g_normal, latent_dim)

    # The slowly add the new layers in
    for i in range(1, len(g_models)):
        # We train both a fade in model and a straight-through model
        [g_normal, g_fadein] = g_models[i]
        [d_normal, d_fadein] = d_models[i]
        [gan_normal, gan_fadein] = gan_models[i]
        #gan_fadein.summary(expand_nested=True, show_trainable=True)
        gen_shape = g_normal.output_shape
        # Scale the dataset to the required new output
        scaled_data = scale_dataset(dataset, gen_shape[1:])
        print(f"Scaled Data: {scaled_data.shape}")

        # Train the fade-in model
        train_epochs(
            g_fadein,
            d_fadein,
            gan_fadein,
            scaled_data,
            e_fadein[i],
            n_batch[i],
            latent_dim,
            True,
        )
        summarize_performance("faded", g_fadein, latent_dim)

        # Train the straight-through model
        train_epochs(
            g_normal,
            d_normal,
            gan_normal,
            scaled_data,
            e_norm[i],
            n_batch[i],
            latent_dim,
        )
        summarize_performance("tuned", g_normal, latent_dim)
