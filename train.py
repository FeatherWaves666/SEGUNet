import os
import os.path as osp
import time
import json
import math
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter

from model.SEG_UNet import SEG_UNet
from model.loss import SLSIoULoss, AverageMeter
from utils.data import IRSTD_Dataset
from utils.metric import ROCMetric, PD_FA, mIoU

def set_seed(seed: int = 3407):
    """Set random seed for reproducibility across PyTorch, NumPy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    try:
        g = torch.Generator()
        g.manual_seed(seed)
    except Exception:
        g = None

    return seed_worker, g


def parse_args():
    parser = ArgumentParser(description='SEG-UNet: Spectral-Edge Gated U-Net for Infrared Small-Target Segmentation')

    # Dataset & Training Hyperparameters (Paper Best Settings)
    parser.add_argument('--dataset-dir', type=str, default='./dataset/IRSTD-1k', help='Path to IRSTD dataset')
    parser.add_argument('--batch-size', type=int, default=8, help='Total batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=500, help='Total training epochs (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--warm-epoch', type=int, default=5, help='Warm-up epochs for deep supervision (default: 5)')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed for reproducibility (default: 3407)')

    # Image Resolution Settings
    parser.add_argument('--base-size', type=int, default=256, help='Base image resolution (default: 256)')
    parser.add_argument('--crop-size', type=int, default=256, help='Crop image resolution (default: 256)')
    
    # Environment & Checkpoints
    parser.add_argument('--multi-gpus', action='store_true', help='Enable Multi-GPU DataParallel training')
    parser.add_argument('--if-checkpoint', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Execution mode')
    parser.add_argument('--weight-path', type=str, default='', help='Path to pretrained model weight for test mode')
    
    # Logging & Experiment Management
    parser.add_argument('--exp-name', type=str, default='SEG_UNet-exp', help='Experiment identifier name')
    parser.add_argument('--save-dir', type=str, default='./experiments', help='Directory to save checkpoints and logs')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='Enable TensorBoard logging')
    parser.add_argument('--save-data', action='store_true', default=True, help='Save metric history to CSV files')

    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, args, exp_id: int = 0):
        self.args = args
        self.mode = args.mode
        self.exp_id = exp_id
        self.start_epoch = 0

        set_worker, g = set_seed(args.seed)

        trainset = IRSTD_Dataset(args, mode='train')
        valset = IRSTD_Dataset(args, mode='val')

        self.train_loader = Data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            worker_init_fn=set_worker,
            generator=g
        )
        self.val_loader = Data.DataLoader(
            valset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            worker_init_fn=set_worker,
            generator=g
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        model = SEG_UNet(input_channels=3)

        if args.multi_gpus and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for parallel training")
            model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        
        model.to(device)
        self.model = model

        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=1e-4)

        # Warm-up + Cosine Annealing Scheduler
        total_epochs = max(1, args.epochs)
        warm_epoch = max(0, args.warm_epoch)

        def lr_lambda(epoch):
            if warm_epoch > 0 and epoch < warm_epoch:
                return float(epoch + 1) / float(warm_epoch)
            else:
                T = max(1, total_epochs - warm_epoch)
                t = max(0, epoch - warm_epoch)
                return 0.5 * (1.0 + math.cos(math.pi * float(t) / float(T)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.down = nn.MaxPool2d(2, 2)
        self.loss_fun = SLSIoULoss()
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC = ROCMetric(1, 10)

        self.best_iou = 0.0
        self.warm_epoch = args.warm_epoch

        self.train_losses = []
        self.val_ious = []
        self.val_pds = []
        self.val_fas = []

        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

        if args.mode == 'train':
            self.main_folder = osp.join(args.save_dir, f"{args.exp_name}_{timestamp}_bs{args.batch_size}_lr{args.lr}")
            os.makedirs(self.main_folder, exist_ok=True)
            self.save_folder = osp.join(self.main_folder, f"exp{exp_id+1:02d}")
            os.makedirs(self.save_folder, exist_ok=True)

            self.save_config()

            if args.tensorboard:
                tb_dir = osp.join(self.main_folder, 'tensorboard', f'exp{exp_id+1:02d}')
                os.makedirs(tb_dir, exist_ok=True)
                self.writer = SummaryWriter(tb_dir)
            else:
                self.writer = None
        else:
            self.save_folder = osp.dirname(args.weight_path) if args.weight_path else os.getcwd()
            os.makedirs(self.save_folder, exist_ok=True)
            if args.weight_path and osp.exists(args.weight_path):
                ckpt = torch.load(args.weight_path, map_location=device)
                state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
                self.model.load_state_dict(state_dict)

    def save_config(self):
        config = vars(self.args)
        config['exp_id'] = self.exp_id
        config['timestamp'] = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        config['device'] = str(self.device)
        with open(osp.join(self.save_folder, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    def save_checkpoint(self, epoch: int, name: str = 'model_checkpoint', iou: float = None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iou': self.best_iou
        }
        filepath = osp.join(self.save_folder, f'{name}.pth')
        try:
            torch.save(state, filepath)
            log_iou = float(iou) if iou is not None else float(self.best_iou)
            with open(osp.join(self.save_folder, 'save_log.txt'), 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Saved {name} epoch={epoch} iou={log_iou:.6f}\n")
        except Exception as e:
            print(f"Warning: Failed to save checkpoint {filepath}: {e}")

    def plot_metrics(self):
        try:
            epochs = list(range(len(self.train_losses)))
            if not epochs:
                return

            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(epochs, self.train_losses, label='Train Loss', color='blue')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 1, 2)
            val_epochs = list(range(len(self.val_ious)))
            if val_epochs:
                plt.plot(val_epochs, self.val_ious, label='IoU', color='red')
                plt.plot(val_epochs, self.val_pds, label='PD', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            plt.legend()
            plt.grid(True)

            img_path = osp.join(self.save_folder, 'metrics.png')
            plt.tight_layout()
            plt.savefig(img_path)
            plt.close()

            if getattr(self.args, 'save_data', False):
                csv_path = osp.join(self.save_folder, 'metrics.csv')
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['epoch', 'train_loss', 'val_iou', 'val_pd', 'val_fa'])
                    for i in range(max(len(self.train_losses), len(self.val_ious))):
                        writer.writerow([
                            i,
                            self.train_losses[i] if i < len(self.train_losses) else '',
                            self.val_ious[i] if i < len(self.val_ious) else '',
                            self.val_pds[i] if i < len(self.val_pds) else '',
                            self.val_fas[i] if i < len(self.val_fas) else ''
                        ])
        except Exception as e:
            print(f"Warning: plot_metrics failed: {e}")

    def train(self, epoch: int):
        self.model.train()
        tbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
        losses = AverageMeter()
        warm_tag = (epoch > self.warm_epoch)

        current_lr = self.optimizer.param_groups[0]['lr']
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.add_scalar('LR/epoch_lr', current_lr, epoch)

        for i, (data, mask) in enumerate(tbar):
            data = data.to(self.device)
            labels = mask.to(self.device)

            masks, pred = self.model(data, warm_tag)
            loss = self.loss_fun(pred, labels, self.warm_epoch, epoch)

            for j in range(len(masks)):
                if j > 0:
                    labels = self.down(labels)
                loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch)

            loss = loss / (len(masks) + 1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), pred.size(0))
            tbar.set_description(f"Epoch {epoch} | Loss: {losses.avg:.4f} | LR: {current_lr:.6f}")

        self.scheduler.step()
        self.train_losses.append(losses.avg)
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.add_scalar('Loss/train', losses.avg, epoch)

    def test(self, epoch: int):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        self.ROC.reset()
        tbar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch}")
        warm_tag = (epoch > self.warm_epoch)

        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):
                data = data.to(self.device)
                mask = mask.to(self.device)

                _, pred = self.model(data, warm_tag)

                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)

                pixAcc, mean_IoU = self.mIoU.get()
                tbar.set_description(f"Validation | IoU: {mean_IoU:.4f}")

            FA, PD = self.PD_FA.get(len(self.val_loader))
            pixAcc, mean_IoU = self.mIoU.get()

            self.val_ious.append(mean_IoU)
            self.val_pds.append(PD[0])
            self.val_fas.append(FA[0] * 1e6)

            if self.mode == 'train':
                if hasattr(self, 'writer') and self.writer is not None:
                    self.writer.add_scalar('Metrics/IoU', mean_IoU, epoch)
                    self.writer.add_scalar('Metrics/PD', PD[0], epoch)
                    self.writer.add_scalar('Metrics/FA', FA[0] * 1e6, epoch)

                if mean_IoU > self.best_iou:
                    self.best_iou = float(mean_IoU)
                    self.save_checkpoint(epoch, 'model_best_iou', iou=mean_IoU)

            elif self.mode == 'test':
                print(f"\n--- SEG-UNet Evaluation Results ---")
                print(f"mIoU : {mean_IoU * 100:.2f}%")
                print(f"Pd   : {PD[0] * 100:.2f}%")
                print(f"Fa   : {FA[0] * 1e6:.2f} (x 10^-6)")


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\nStarting SEG-UNet Training / Testing Pipeline")
    print(f"Parameters: LR={args.lr}, BatchSize={args.batch_size}, Seed={args.seed}, WarmEpoch={args.warm_epoch}, Epochs={args.epochs}\n")

    trainer = Trainer(args, exp_id=0)

    if args.mode == 'train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)

        trainer.plot_metrics()
        trainer.save_checkpoint(args.epochs - 1, 'model_last')
        print(f"\nTraining completed! Best IoU: {trainer.best_iou:.4f}")
    else:
        trainer.test(epoch=0)

    if hasattr(trainer, 'writer') and trainer.writer is not None:
        trainer.writer.close()
