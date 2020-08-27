import tensorflow as tf
class TFMLPBase(tf.keras.model):
   def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.masking = tf.keras.layers.Masking(mask_value=0)
        
        self.mlp = Dense(units=8, activation='relu')
        self.c_proj = tf.keras.layers.Dense(
                1, activation='sigmoid',
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
            )

    def call(self, x, training=False):
        x = self.masking(x)
        bz, sl = shape_list(x)[:2]
        x = x.reshape(bz*sl,:)
        x = self.mlp(x)
        out = self.c_proj(x)

        return out
        