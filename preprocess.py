import os, re, random
import numpy as np
import rasterio
from pathlib import Path

S2_BAND_ORDER = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]

def band_tag(pathname: str):
    name = os.path.basename(pathname).upper()
    m = re.search(r"_B([0-9]{1,2}|8A)\b", name) or re.search(r"\bB([0-9]{1,2}|8A)\b", name)
    if not m: return None
    raw = m.group(1)
    if raw.endswith("A"): return "B8A"
    else: return f"B{int(raw):02d}"

def read_single_band_stack(folder: str,
                           desired_tags=S2_BAND_ORDER,
                           strict=False, fill_missing=True, verbose=False):
    files = sorted(list(Path(folder).glob("*.tif")))
    tagged = {}
    for f in files:
        t = band_tag(str(f))
        if t: tagged[t] = str(f)

    ref = None
    for t in desired_tags:
        if t in tagged: ref = tagged[t]; break
    if ref is None: raise RuntimeError(f"No bands in {folder}")
    with rasterio.open(ref) as src0:
        H,W = src0.height, src0.width
        meta = src0.meta.copy()
        zero = np.zeros((H,W), np.float32)

    arrs = []
    for t in desired_tags:
        if t in tagged:
            with rasterio.open(tagged[t]) as s: arrs.append(s.read(1))
        else:
            if strict and not fill_missing: raise RuntimeError(f"Missing {t}")
            arrs.append(zero.copy())
    arr = np.stack(arrs, axis=-1)
    return arr, meta

def read_city_pair_13bands(images_root, labels_root, city):
    t1_dir = f"{images_root}/{city}/imgs_1_rect"
    t2_dir = f"{images_root}/{city}/imgs_2_rect"
    lab    = f"{labels_root}/{city}/cm/{city}-cm.tif"
    t1,_ = read_single_band_stack(t1_dir)
    t2,_ = read_single_band_stack(t2_dir)
    with rasterio.open(lab) as s:
        mask = s.read(1)
        if mask.max()>1: mask=(mask>(mask.max()/2)).astype(np.uint8)
    return t1,t2,mask,_

def norm_p02_p98(x):
    x = x.astype(np.float32)
    q1 = np.percentile(x, 2, axis=(0,1), keepdims=True)
    q2 = np.percentile(x, 98, axis=(0,1), keepdims=True)
    x = (x-q1)/np.clip(q2-q1,1e-6,None)
    return np.clip(x,0,1)

def same_tiles_indices(L, k, s):
    if L<=k: return [0]
    idx=list(range(0,L,s))
    if idx[-1]!=L-k: idx.append(L-k)
    return idx

def tile_city_save(t1,t2,m,out_dir,city,tile, stride, skip_empty_prob=0.8):
    out=Path(out_dir)/"patches"; out.mkdir(parents=True,exist_ok=True)
    H,W,_=t1.shape
    ys=same_tiles_indices(H,tile,stride)
    xs=same_tiles_indices(W,tile,stride)
    saved=0
    for y in ys:
        for x in xs:
            y2=min(y+tile,H); x2=min(x+tile,W)
            a1=t1[y:y2,x:x2]; a2=t2[y:y2,x:x2]; am=m[y:y2,x:x2]
            ph,pw=tile-a1.shape[0], tile-a1.shape[1]
            if ph>0 or pw>0:
                a1=np.pad(a1,((0,ph),(0,pw),(0,0)))
                a2=np.pad(a2,((0,ph),(0,pw),(0,0)))
                am=np.pad(am,((0,ph),(0,pw)))
            if am.sum()==0 and random.random()<skip_empty_prob: continue
            a1n=norm_p02_p98(a1); a2n=norm_p02_p98(a2)
            base=f"{city}_{y:05d}_{x:05d}"
            np.save(out/f"{base}_t1.npy",a1n.astype(np.float32))
            np.save(out/f"{base}_t2.npy",a2n.astype(np.float32))
            np.save(out/f"{base}_mask.npy",(am>0).astype(np.uint8))
            saved+=1
    return saved
