from .layers import Attention
from .utils import create_attention_mask

import tensorflow as tf
import numpy as np


class AkiLstm(tf.keras.Model):
    def __init__(self, n_features=16, timesteps=48, **kwargs):
        super().__init__(**kwargs)
        self.attn = Attention(timesteps=timesteps)
        self.masking = tf.keras.layers.Masking(mask_value=0)
        self.lstm = tf.keras.layers.LSTM(256, return_sequences=True)
        self.dist = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, activation='sigmoid')
        )

    def call(self, x, training=False):
        mask = create_attention_mask(x)
        x, w = self.attn(x, mask=mask, training=training)
        x = self.masking(x)
        x = self.lstm(x)
        x = self.dist(x)

        return x, w
