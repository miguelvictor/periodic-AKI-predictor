import tensorflow as tf


def create_attention_mask(x):
    '''
    Accepts an input of shape: (batch_size, timesteps, features).
    Timesteps whose features are all 0 will be treated as padding.
    Returns an output of shape: (batch_size, timesteps, 1).
    '''
    x = tf.reduce_any(x != 0, axis=-1)
    x = x[:, :, tf.newaxis]
    x = tf.cast(x, tf.float32)
    return (1 - x) * -1e9
