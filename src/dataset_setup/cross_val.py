import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision.transforms import functional as TF
from sklearn.model_selection import KFold
from scipy.io import loadmat
from collections import defaultdict


# --- Configuration ---
BATCH_SIZE = 4
SEED = 42


# --- Helper Functions ---
def _norm_stem(name):
    """Normalize filenames to match images with labels (removes suffixes like _mask, _label)."""
    
    s = os.path.splitext(name)[0].lower()
    for suf in ("_label", "_labels", "_mask", "-label", "-labels", "-mask"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def load_label_arr(path):
    """Load label mask from .mat or image files."""

    if str(path).lower().endswith(".mat"):
        m = loadmat(path)
        # Check common keys for mask data
        for k in ("type_map", "type", "label", "gt", "inst_map", "inst"):
            if k in m:
                arr = np.asarray(m[k]).squeeze().astype(np.int32)
                # If instance map, convert to binary (0/1) or keep as is depending on need
                # Here assuming we want class values. If 'inst', usually just binary FG/BG.
                if "inst" in k: 
                    return (arr > 0).astype(np.int32)
                return arr
        raise ValueError(f"No usable keys in {path}")
    else:
        # Load standard image mask
        return np.array(Image.open(path).convert("L")).astype(np.int32)


def collect_label_values(labels_dir):
    """Scans the label directory to find all unique class IDs."""

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".mat")
    vals = set()
    print(f"Scanning {labels_dir} for label values...")
    for f in sorted(os.listdir(labels_dir)):
        if not f.lower().endswith(exts):
            continue
        arr = load_label_arr(os.path.join(labels_dir, f))
        vals.update(np.unique(arr).tolist())
    
    vals = sorted(int(v) for v in vals)
    val2idx = {v: i for i, v in enumerate(vals)}
    return vals, val2idx



# --- Generic Dataset Class ---
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, img_subdir="Images", lbl_subdir="Labels", val2idx=None, transform=None):
        self.root = Path(root_dir)
        self.images_dir = self.root / img_subdir
        self.labels_dir = self.root / lbl_subdir
        self.transform = transform
        
        # If val2idx is not provided, we calculate it (expensive) or assume default
        if val2idx is None:
            _, self.val2idx = collect_label_values(self.labels_dir)
        else:
            self.val2idx = val2idx

        # File extensions to look for
        img_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        lbl_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".mat")

        # Find files
        imgs = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith(img_exts)])
        lbls = sorted([f for f in os.listdir(self.labels_dir) if f.lower().endswith(lbl_exts)])

        # Map labels by normalized stem
        lbl_map = {_norm_stem(lf): lf for lf in lbls}
        
        self.pairs = []
        for imf in imgs:
            k = _norm_stem(imf)
            if k in lbl_map:
                self.pairs.append((
                    os.path.join(self.images_dir, imf), 
                    os.path.join(self.labels_dir, lbl_map[k])
                ))

        # Fallback for matching by index if names don't match (dangerous but sometimes needed)
        if len(self.pairs) == 0 and len(imgs) > 0 and len(lbls) > 0:
            print("Warning: No filename matches found. Falling back to index matching.")
            m = min(len(imgs), len(lbls))
            self.pairs = [(
                os.path.join(self.images_dir, imgs[i]), 
                os.path.join(self.labels_dir, lbls[i])
            ) for i in range(m)]

        print(f"Dataset [{self.root.name}]: Found {len(self.pairs)} pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lab_path = self.pairs[idx]
        
        # Load Image
        x_pil = Image.open(img_path).convert("RGB")
        
        # Load Label
        arr = load_label_arr(lab_path)
        
        # Map raw label values to 0, 1, 2... class indices
        idx_arr = np.vectorize(self.val2idx.get, otypes=[np.int64])(arr)

        if self.transform:
            x_np = np.asarray(x_pil)
            y_np = idx_arr.astype(np.int64)
            out = self.transform(image=x_np, mask=y_np)
            return out["image"], out["mask"].long()

        # Default Transform (ToTensor + Normalize)
        x = TF.to_tensor(x_pil)
        x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        y = torch.from_numpy(idx_arr.astype(np.int64))
        return x, y



# --- Subset Wrapper (Preserves Transform Logic) ---
class SubsetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.base_dataset = subset.dataset
        
        # Handle nested subsets if necessary
        while isinstance(self.base_dataset, Subset):
            self.base_dataset = self.base_dataset.dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        original_idx = self.subset.indices[idx]
        # Access the pair directly from the base dataset
        img_path, lab_path = self.base_dataset.pairs[original_idx]
        
        x_pil = Image.open(img_path).convert("RGB")
        arr = load_label_arr(lab_path)
        
        idx_arr = np.vectorize(
            self.base_dataset.val2idx.get, otypes=[np.int64]
        )(arr)
        
        x_np = np.asarray(x_pil)
        y_np = idx_arr.astype(np.int64)

        if self.transform:
            out = self.transform(image=x_np, mask=y_np)
            return out["image"], out["mask"].long()

        x = TF.to_tensor(x_pil)
        x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        y = torch.from_numpy(y_np)
        return x, y



# --- Main Setup Function ---
def setup_experiment(
    data_root, 
    img_subdir="Images", 
    lbl_subdir="Labels", 
    test_ratio=0.2, 
    n_folds=5
):
    """
    Generalizes the setup for any dataset.
    1. Scans labels to determine classes.
    2. Creates the full dataset.
    3. Splits into Test (hold-out) and CV (train/val) sets.
    4. Returns the CV indices/subset and the Test Loader.
    """

    data_path = Path(data_root)
    lbl_path = data_path / lbl_subdir
    
    # 1. Auto-detect classes
    vals, val2idx = collect_label_values(lbl_path)
    print(f"Classes detected: {vals}")
    
    # 2. Create Full Dataset
    full_ds = SegmentationDataset(
        data_path, 
        img_subdir=img_subdir, 
        lbl_subdir=lbl_subdir, 
        val2idx=val2idx
    )

    # 3. Create Split Indices
    num_total = len(full_ds)
    num_test = int(test_ratio * num_total)
    
    indices = np.arange(num_total)
    np.random.seed(SEED)
    np.random.shuffle(indices)
    
    test_idx = indices[:num_test]
    cv_idx = indices[num_test:]
    
    # 4. Create Subsets
    test_subset = Subset(full_ds, test_idx)
    cv_subset = Subset(full_ds, cv_idx) # This will be split further in K-Fold
    
    # 5. Create Test Loader (Fixed Hold-out set)
    # Wrap test_subset to apply validation transforms if needed
    test_ds_wrapped = SubsetWrapper(test_subset, transform=None) 
    test_loader = DataLoader(
        test_ds_wrapped, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2
    )

    print(f"\n--- Experiment Split Summary for {data_path.name} ---")
    print(f"Total Images: {num_total}")
    print(f"Test Set    : {len(test_subset)} (Ratio: {test_ratio})")
    print(f"CV Set      : {len(cv_subset)} (To be split into {n_folds} folds)")
    
    return cv_subset, test_loader, n_folds, val2idx