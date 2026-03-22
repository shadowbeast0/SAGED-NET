import os
import glob
import re
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF


# --- 1. Helper Functions ---
def natsort_key(s):
    """Natural sort key for filenames (e.g., img1, img2, ... img10)."""

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def find_pairs(folder_path, img_subdir="images", mask_subdir="masks"):
    """
    Scans a folder for 'images' and 'masks' subdirectories and pairs files.
    Returns: (sorted_img_paths, sorted_mask_paths)
    """

    folder = Path(folder_path)
    img_dir = folder / img_subdir
    mask_dir = folder / mask_subdir
    
    if not img_dir.exists() or not mask_dir.exists():
        return [], []

    # Supported extensions
    exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
    
    # Gather files
    imgs = []
    masks = []
    for ext in exts:
        imgs.extend(img_dir.glob(ext))
        masks.extend(mask_dir.glob(ext))
    
    # Sort to ensure alignment
    imgs = sorted([str(p) for p in imgs], key=natsort_key)
    masks = sorted([str(p) for p in masks], key=natsort_key)

    # Sanity Check
    if len(imgs) != len(masks):
        print(f"Warning in {folder_path}: Found {len(imgs)} images but {len(masks)} masks!")
    
    # Truncate to matching length to prevent crashes
    n = min(len(imgs), len(masks))
    return imgs[:n], masks[:n]


def scan_for_classes(mask_paths_list):
    """
    Scans a list of mask paths to find all unique pixel values (classes).
    Returns: (values_list, val2idx_dict, num_classes)
    """

    print("Scanning masks for classes... (this may take a moment)")
    all_vals = set()
    
    # Optimization: If dataset is huge, maybe scan only first 500 masks
    # or use a random sample. For now, we scan all for accuracy.
    for p in mask_paths_list:
        # Load as 'L' (8-bit pixels, black and white)
        a = np.array(Image.open(p).convert('L'))
        unique = np.unique(a)
        all_vals.update(unique.tolist())
        
    vals = sorted(list(all_vals))
    
    # Binary check: if only 0 and 255 found, map them to 0 and 1
    if set(vals) == {0, 255}:
        vals = [0, 255]
        
    val2idx = {v: i for i, v in enumerate(vals)}
    return vals, val2idx, len(vals)



# --- 2. Generic Dataset Class ---
class SegmentationDataset(Dataset):
    def __init__(self, img_paths, mask_paths, size, val2idx, training=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.size = size
        self.val2idx = val2idx
        self.training = training
        
    def __len__(self):
        return len(self.img_paths)
        
    def _augment(self, x, y):
        # 50% Horizontal Flip
        if random.random() < 0.5:
            x = TF.hflip(x)
            y = TF.hflip(y)
            
        # 50% Vertical Flip
        if random.random() < 0.5:
            x = TF.vflip(x)
            y = TF.vflip(y)
            
        # 50% Random Rotation (0, 90, 180, 270)
        if random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            x = TF.rotate(x, angle)
            y = TF.rotate(y, angle, fill=0) # Fill 0 for background
            
        return x, y
        
    def __getitem__(self, idx):
        # 1. Load Data
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        x = Image.open(img_path).convert('RGB')
        y = Image.open(mask_path).convert('L') # Load as grayscale index map
        
        # 2. Resize
        # Images use Bilinear (smooth), Masks use Nearest (preserve exact class integers)
        x = x.resize((self.size, self.size), Image.BILINEAR)
        y = y.resize((self.size, self.size), Image.NEAREST)
        
        # 3. Augment (only if training)
        if self.training:
            x, y = self._augment(x, y)
            
        # 4. To Tensor & Normalize
        x = TF.to_tensor(x) # Scales [0, 255] -> [0.0, 1.0]
        x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # 5. Map Mask Values to [0, 1, 2...]
        y_np = np.array(y, dtype=np.uint8)
        # Efficient vectorization to map raw pixel values (e.g., 0, 255) to class indices (e.g., 0, 1)
        y_mapped = np.vectorize(self.val2idx.get)(y_np)
        y = torch.from_numpy(y_mapped).long()
        
        return x, y



# --- 3. Main Setup Function ---
def setup_segmentation_loaders(data_root, img_size=256, batch_size=8):
    """
    General setup function. 
    Expects structure:
      root/
        train/ (images/, masks/)
        validate/ (images/, masks/)  <-- Optional, splits into Val/Test if test is missing
        test/ (images/, masks/)      <-- Optional
    """
    
    root = Path(data_root)
    
    # A. Find File Paths
    train_imgs, train_msks = find_pairs(root / "train")
    val_imgs, val_msks = find_pairs(root / "validate") # Check for 'validate'
    if not val_imgs: 
        val_imgs, val_msks = find_pairs(root / "val")  # Check for 'val'

    test_imgs, test_msks = find_pairs(root / "test")

    # B. Handle Missing Test Set (Split Validation if needed)
    if not test_imgs and len(val_imgs) > 0:
        print("No explicit 'test' folder found. Splitting 'validate' into Val/Test (50/50).")
        n_val = len(val_imgs) // 2
        # First half for Val, second half for Test
        test_imgs, test_msks = val_imgs[n_val:], val_msks[n_val:]
        val_imgs, val_msks = val_imgs[:n_val], val_msks[:n_val]
    
    # Error Handling
    if not train_imgs:
        raise ValueError(f"No training data found in {root}/train")

    # C. Scan for Class Mapping (using ALL masks to be safe)
    all_mask_paths = train_msks + val_msks + test_msks
    vals, val2idx, num_classes = scan_for_classes(all_mask_paths)
    
    print(f"Found {num_classes} classes. Values: {vals} -> Mapped to: {list(val2idx.values())}")

    # D. Create Datasets
    train_ds = SegmentationDataset(train_imgs, train_msks, img_size, val2idx, training=True)
    val_ds = SegmentationDataset(val_imgs, val_msks, img_size, val2idx, training=False)
    test_ds = SegmentationDataset(test_imgs, test_msks, img_size, val2idx, training=False)

    # E. Create Loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # F. Summary
    print("-" * 30)
    print(f"Data Root: {root}")
    print(f"Train: {len(train_ds)} samples ({len(train_dl)} batches)")
    print(f"Val  : {len(val_ds)} samples ({len(val_dl)} batches)")
    print(f"Test : {len(test_ds)} samples ({len(test_dl)} batches)")
    print("-" * 30)

    return train_dl, val_dl, test_dl, num_classes