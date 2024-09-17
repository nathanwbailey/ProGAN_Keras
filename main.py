"""Main Entry File to Train the ProGAN."""

from discriminator import define_discriminator
from generator import define_generator
from helper_functions import load_real_samples
from network import define_composite_model
from train import train

COMPRESSED_DATA_STR = "faces_dataset.npz"
N_BLOCKS = 6
LATENT_DIM = 100
N_BATCH = [16, 16, 16, 8, 4, 4]
N_EPOCHS = [5, 8, 8, 10, 10, 10]

D_MODELS = define_discriminator(N_BLOCKS)
G_MODELS = define_generator(LATENT_DIM, N_BLOCKS)
GAN_MODELS = define_composite_model(D_MODELS, G_MODELS)

DATASET = load_real_samples(COMPRESSED_DATA_STR)
print(f"Loaded: {DATASET.shape}")

train(
    G_MODELS,
    D_MODELS,
    GAN_MODELS,
    DATASET,
    LATENT_DIM,
    N_EPOCHS,
    N_EPOCHS,
    N_BATCH,
)
