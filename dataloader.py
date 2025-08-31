import numpy as np, tensorflow as tf
from pathlib import Path

def list_tile_bases(root):
    t1s=sorted(Path(root).glob("patches/*_t1.npy"))
    return [str(p).replace("_t1.npy","") for p in t1s]

def load_triplet_np(base):
    a1=np.load(base+"_t1.npy")
    a2=np.load(base+"_t2.npy")
    m =np.load(base+"_mask.npy")
    m =np.expand_dims(m,-1).astype(np.float32)
    return a1.astype(np.float32),a2.astype(np.float32),m

def tf_load(base, tile_size):
    a1,a2,m=tf.numpy_function(
        lambda b: load_triplet_np(b.decode()),
        [base],[tf.float32,tf.float32,tf.float32]
    )
    a1.set_shape([tile_size,tile_size,13]); a2.set_shape([tile_size,tile_size,13]); m.set_shape([tile_size,tile_size,1])
    return (a1,a2),m

@tf.function
def aug_pair(a1,a2,m):
    if tf.random.uniform([])<0.5:
        a1=tf.reverse(a1,[1]);a2=tf.reverse(a2,[1]);m=tf.reverse(m,[1])
    if tf.random.uniform([])<0.5:
        a1=tf.reverse(a1,[0]);a2=tf.reverse(a2,[0]);m=tf.reverse(m,[0])
    k=tf.random.uniform([],0,4,dtype=tf.int32)
    a1=tf.image.rot90(a1,k);a2=tf.image.rot90(a2,k);m=tf.image.rot90(m,k)
    return (a1,a2),m

def make_ds(bases,augment=False,shuffle=True,batch_size=8, tile_size=128):
    ds=tf.data.Dataset.from_tensor_slices(bases)
    if shuffle: ds=ds.shuffle(buffer_size=min(4096,len(bases)))
    ds=ds.map(lambda b: tf_load(b,tile_size),num_parallel_calls=tf.data.AUTOTUNE)
    if augment: ds=ds.map(lambda x,y: aug_pair(x[0],x[1],y),num_parallel_calls=tf.data.AUTOTUNE)
    ds=ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
