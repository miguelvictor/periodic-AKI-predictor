import tensorflow as tf


class TFAkiBase(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.masking = tf.keras.layers.Masking(mask_value=0)
        self.lstm = tf.keras.layers.LSTM(256, return_sequences=True)
        self.dist = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                1, activation='sigmoid',
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
            )
        )

    def call(self, x, training=False):
        x = self.masking(x)
        x = self.lstm(x)
        x = self.dist(x)

        return x
