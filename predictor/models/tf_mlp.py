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
        mask = tf.reduce_any(x!=0,axis=-1)
        x = tf.boolean_mask(x,mask)
        x = self.mlp(x)
        out = self.c_proj(x)
        return out
        