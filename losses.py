import tensorflow as tf
from tensorflow import keras

def dice_loss(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
    inter  = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])    # (B,)
    denom  = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])    # (B,)
    dice   = (2. * inter + eps) / (denom + eps)                # (B,)
    return 1.0 - dice                                          # (B,)

def bce_dice(y_true, y_pred, alpha=0.5):
    # Per-pixel BCE map: same rank as y_true (usually 4)
    bce_map = keras.backend.binary_crossentropy(y_true, y_pred)  # (B, ..., ...)
    # Reduce over all axes except batch, without Python conditionals:
    reduce_axes = tf.range(1, tf.rank(bce_map))                  # e.g., [1,2,3]
    bce = tf.reduce_mean(bce_map, axis=reduce_axes)              # (B,)

    d = dice_loss(y_true, y_pred)                                # (B,)
    loss_per_sample = alpha * bce + (1.0 - alpha) * d            # (B,)
    return tf.reduce_mean(loss_per_sample)    

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1. - keras.backend.epsilon())
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    p_t = y_true*y_pred + (1-y_true)*(1-y_pred)
    mod = tf.pow(1. - p_t, gamma)
    alpha_factor = y_true*alpha + (1-y_true)*(1-alpha)
    return tf.reduce_mean(alpha_factor * mod * bce)
