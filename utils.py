import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def iou_metric(y_true, y_pred, thr=0.5, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > thr, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3]) - inter
    return tf.reduce_mean((inter + eps) / (union + eps))

def f1_metric(y_true, y_pred, thr=0.5, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > thr, tf.float32)
    tp = tf.reduce_sum(y_true*y_pred, axis=[1,2,3])
    fp = tf.reduce_sum((1-y_true)*y_pred, axis=[1,2,3])
    fn = tf.reduce_sum(y_true*(1-y_pred), axis=[1,2,3])
    return tf.reduce_mean((2*tp + eps)/(2*tp + fp + fn + eps))

def visualize_preds(model, dataset, n=3, thr=0.5):
    # Get one batch
    (x1, x2), y = next(iter(dataset))
    # Predict probabilities
    probs = model.predict([x1, x2], verbose=0)  # (B,H,W,1)

    to_np = lambda t: t.numpy() if hasattr(t, "numpy") else t

    B = probs.shape[0]
    nshow = int(min(n, B))

    for i in range(nshow):
        # Build RGB (Sentinel-2 RGB = B04,B03,B02 => indices 3,2,1)
        if x1.shape[-1] >= 4:
            # use tf.gather on tensors, then convert to numpy
            rgb1 = to_np(tf.gather(x1[i], [3, 2, 1], axis=-1))
            rgb2 = to_np(tf.gather(x2[i], [3, 2, 1], axis=-1))
        else:
            # fallback to first 3 channels
            rgb1 = to_np(x1[i, ..., :3])
            rgb2 = to_np(x2[i, ..., :3])

        rgb1 = np.clip(rgb1, 0, 1)
        rgb2 = np.clip(rgb2, 0, 1)

        gt = to_np(y[i])
        if gt.ndim == 3:  # (H,W,1) -> (H,W)
            gt = gt[..., 0]

        pr_prob = probs[i, ..., 0]
        pr_bin  = (pr_prob >= thr).astype(float)

        fig, ax = plt.subplots(1, 4, figsize=(14, 4))
        ax[0].imshow(rgb1); ax[0].set_title("T1 RGB"); ax[0].axis('off')
        ax[1].imshow(rgb2); ax[1].set_title("T2 RGB"); ax[1].axis('off')
        ax[2].imshow(gt, cmap='gray', vmin=0, vmax=1); ax[2].set_title("GT"); ax[2].axis('off')
        ax[3].imshow(pr_bin, cmap='gray', vmin=0, vmax=1); ax[3].set_title(f"Pred (t={thr})"); ax[3].axis('off')
        plt.tight_layout()
        plt.show()
