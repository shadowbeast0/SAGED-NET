import albumentations as A
from albumentations.pytorch import ToTensorV2

img_size = 512

train_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent={"x": 0.05, "y": 0.05}, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
    A.ElasticTransform(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], is_check_shapes=False)

val_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], is_check_shapes=False)

test_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], is_check_shapes=False)