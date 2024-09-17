import matplotlib.pyplot as plt
import numpy as np
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
    gan_model,
    dataset,
    n_epochs: int,
    n_batch: int,
    latent_dim: int,
    fadein: bool = False,
) -> None:
    batch_per_epoch = int(dataset.shape[0] / n_batch)
    n_steps = batch_per_epoch * n_epochs

    half_batch = int(n_batch / 2)

    print(f"Number of Steps: {n_steps}")

    for i in range(n_steps):
        if fadein:
            update_fadein([g_model, d_model, gan_model], i, n_steps)

        x_real, y_real = generate_real_samples(dataset, half_batch)
        x_fake, y_fake = generate_fake_samples(
            g_model, latent_dim, half_batch
        )

        d_loss1 = d_model.train_on_batch(x_real, y_real)
        d_loss2 = d_model.train_on_batch(x_fake, y_fake)

        z_input = generate_latent_points(latent_dim, n_batch)
        y_real2 = np.ones((n_batch, 1))

        g_loss = gan_model.train_on_batch(z_input, y_real2)[0]

        print(
            f"Batch: {i+1}, d_loss1: {d_loss1:4f}, d_loss2: {d_loss2:4f}, g_loss: {g_loss:4f}"
        )


def train(
    g_models,
    d_models,
    gan_models,
    dataset,
    latent_dim,
    e_norm,
    e_fadein,
    n_batch,
) -> None:
    g_normal, d_normal, gan_normal = (
        g_models[0][0],
        d_models[0][0],
        gan_models[0][0],
    )
    gen_shape = g_normal.output_shape

    scaled_data = scale_dataset(dataset, gen_shape[1:])
    print(f"Scaled Data: {scaled_data.shape}")

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

    for i in range(1, len(g_models)):
        [g_normal, g_fadein] = g_models[i]
        [d_normal, d_fadein] = d_models[i]
        [gan_normal, gan_fadein] = gan_models[i]

        gen_shape = g_normal.output_shape
        scaled_data = scale_dataset(dataset, gen_shape[1:])
        print(f"Scaled Data: {scaled_data.shape}")

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
