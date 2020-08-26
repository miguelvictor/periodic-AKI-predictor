import tensorflow as tf


def create_attention_mask(x):
    x = tf.reduce_any(x != 0, axis=-1)
    x = x[:, :, tf.newaxis]
    x = tf.cast(x, tf.float32)
    return (1 - x) * -1e9


class Attention(tf.keras.layers.Layer):
    def __init__(self, n_features=16, **kwargs):
        super().__init__(**kwargs)

        self.proj = tf.keras.layers.Dense(n_features)
        self.drop = tf.keras.layers.Dropout(0.1)

    def call(self, x, training=False):
        w = self.proj(x)
        w = self.drop(w, training=training)

        mask = create_attention_mask(x)
        w = tf.nn.softmax(w + mask, axis=-2)

        return tf.multiply(x, w), w


class TFAkiLstm(tf.keras.Model):
    def __init__(self, timesteps=8, n_features=16, **kwargs):
        super().__init__(**kwargs)

        self.norm = tf.keras.layers.BatchNormalization()
        self.attn = Attention(n_features)
        self.masking = tf.keras.layers.Masking(mask_value=0)
        self.lstm = tf.keras.layers.LSTM(256, return_sequences=True)
        self.dist = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                1,
                activation='sigmoid',
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
            )
        )

    def call(self, x, training=False):
        x = self.norm(x)
        x, w = self.attn(x, training=training)
        x = self.masking(x)
        x = self.lstm(x)
        x = self.dist(x)

        return x, w
