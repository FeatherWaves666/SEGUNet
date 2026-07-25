import os
import os.path as osp
import random
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter

class IRSTD_Dataset(Data.Dataset):
    """
    Dataset loader for Infrared Small-Target Detection (IRSTD) benchmarks.
    Compatible with IRSTD-1k, NUAA-SIRST, NUDT-SIRST, and custom datasets.
    
    Expected folder structure:
        dataset_dir/
        ├── trainval.txt (or train.txt)
        ├── test.txt
        ├── images/
        └── masks/
    """
    def __init__(self, args, mode: str = 'train'):
        super().__init__()
        dataset_dir = getattr(args, 'dataset_dir', './dataset/IRSTD-1k')
        assert mode in ['train', 'val', 'test'], f"Unsupported mode: {mode}"

        if mode == 'train':
            txtfile = 'trainval.txt' if osp.exists(osp.join(dataset_dir, 'trainval.txt')) else 'train.txt'
        else:
            txtfile = 'test.txt'

        self.list_dir = osp.join(dataset_dir, txtfile)
        self.imgs_dir = osp.join(dataset_dir, 'images')
        self.label_dir = osp.join(dataset_dir, 'masks')

        self.names = []
        if osp.exists(self.list_dir):
            with open(self.list_dir, 'r') as f:
                self.names = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # Fallback: scan images directory if split file is missing
            if osp.exists(self.imgs_dir):
                self.names = [osp.splitext(f)[0] for f in os.listdir(self.imgs_dir) if f.endswith(('.png', '.jpg', '.bmp'))]

        self.mode = mode
        self.crop_size = getattr(args, 'crop_size', 256)
        self.base_size = getattr(args, 'base_size', 256)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '.png')

        # Fallback extension check if .png doesn't exist
        if not osp.exists(img_path):
            for ext in ['.jpg', '.bmp', '.PNG', '.JPG']:
                if osp.exists(osp.join(self.imgs_dir, name + ext)):
                    img_path = osp.join(self.imgs_dir, name + ext)
                    break

        if not osp.exists(label_path):
            for ext in ['.jpg', '.bmp', '.PNG', '.JPG']:
                if osp.exists(osp.join(self.label_dir, name + ext)):
                    label_path = osp.join(self.label_dir, name + ext)
                    break

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        else:
            img, mask = self._testval_sync_transform(img, mask)

        img_tensor = self.transform(img)
        mask_tensor = transforms.ToTensor()(mask)
        return img_tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.names)

    def _sync_transform(self, img: Image.Image, mask: Image.Image):
        # Random horizontal flip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        crop_size = self.crop_size
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size

        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh

        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # Padding if necessary
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        # Random crop
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # Gaussian blur augmentation
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return img, mask

    def _testval_sync_transform(self, img: Image.Image, mask: Image.Image):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)
        return img, mask