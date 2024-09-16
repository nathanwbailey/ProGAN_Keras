from dataset_helper_functions import load_faces
import numpy as np

COMPRESSED_DATA_STR = 'faces_dataset.npz'
DATA_DIR = 'dataset/'


all_faces = load_faces(directory=DATA_DIR, n_faces=10000)

np.savez_compressed(COMPRESSED_DATA_STR, all_faces)
