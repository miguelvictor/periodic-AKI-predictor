import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        timesteps,
        pdrop=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(timesteps)
        self.drop = tf.keras.layers.Dropout(pdrop)

    def call(self, x, mask=None, training=False):
        w = tf.transpose(x, perm=(0, 2, 1))
        w = self.dense(w)
        w = self.drop(w, training=training)
        w = tf.transpose(w, perm=(0, 2, 1))

        if mask is not None:
            w = w + mask

        # apply softmax on the timesteps axis
        w = tf.nn.softmax(w, axis=-2)
        x = x * w

        return x, w
