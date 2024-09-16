from tensorflow import keras
import tensorflow as tf
from numpy.typing import NDArray
from typing import Any


class PixelNormalization(keras.layers.Layer):
    """Pixel Normalization Layer."""

    def call(self, inputs: NDArray[Any]) -> NDArray[Any]:
        values = inputs**2.0
        # Compute mean across the channel dimension
        # For each pixel, calculate the mean across all channels
        # (H, W, 1) result
        mean_values = tf.math.reduce_mean(values, axis=-1, keepdims=True)
        mean_values += 1.0e-8
        l2 = tf.math.sqrt(mean_values)
        normalized_pixels = inputs / l2
        return normalized_pixels


class MinibatchStdev(keras.layers.Layer):
    def call(self, inputs: NDArray[Any]) -> tf.Tensor:
        # Take the mean across all the images in the batch
        # (1, H, W, C)
        mean = tf.math.reduce_mean(inputs, axis=0, keepdims=True)
        squ_diffs = tf.math.square(inputs - mean)
        # Take the mean across all the images in the batch
        # (1, H, W, C)
        mean_sq_diff = tf.math.reduce_mean(squ_diffs, axis=0, keepdims=True)
        mean_sq_diff += 1e-8
        stdev = tf.math.sqrt(mean_sq_diff)

        # Reduce all dims
        # (1, 1, 1, 1)
        mean_pix = tf.math.reduce_mean(stdev, axis=None, keepdims=True)
        shape = tf.shape(inputs)
        outputs = tf.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        # Add another channel dim
        return tf.concat([inputs, outputs], axis=-1)


class WeightedSum(keras.layers.Add):
    def __init__(self, alpha: float = 0.0):
        super().__init__()
        self.alpha = tf.Variable(alpha, trainable=False)

    def _merge_function(self, inputs: NDArray[Any]) -> NDArray[Any]:
        """Override the merge function of Add Layer."""
        assert len(inputs) == 2
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


def wasserstein_loss(y_true: NDArray[Any], y_pred: NDArray[Any]) -> tf.Tensor:
    return tf.math.reduce_mean(y_true * y_pred)