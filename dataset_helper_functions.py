import os
from typing import Any

import mtcnn
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tensorflow import keras


def load_image(filename: str) -> NDArray[Any]:
    """Load and Convert Image to Numpy Array."""
    image = Image.open(filename)
    image_rgb = image.convert("RGB")
    return np.asarray(image_rgb)


def extract_face(
    model: keras.models.Model,
    pixels: NDArray[Any],
    required_size: tuple[int, int] = (128, 128),
) -> NDArray[Any] | None:
    """Extract the Face from an Image."""
    faces = model.detect_faces(pixels)
    if len(faces) == 0:
        return None

    x1, y1, width, height = faces[0]["box"]
    x1, y1 = abs(x1), abs(y1)

    x2, y2 = x1 + width, y1 + height
    face_pixels = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face_pixels)
    image_resized = image.resize(required_size)
    return np.asarray(image_resized)


def load_faces(directory: str, n_faces: int) -> NDArray[Any]:
    """Load all the faces from a dir of files."""
    model = mtcnn.mtcnn.MTCNN()
    faces: list[NDArray[Any]] = []

    for filename in os.listdir(directory):
        pixels = load_image(directory + filename)
        face = extract_face(model, pixels)
        if face is None:
            continue
        print(f"Found: {len(faces)} faces")
        faces.append(face)
        if len(faces) >= n_faces:
            break

    return np.asarray(faces)
