import numpy as np

COMPRESSED_DATA_STR = 'faces_dataset.npz'
data = np.load(COMPRESSED_DATA_STR)

faces = data['arr_0']
