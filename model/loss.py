import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SoftIoULoss(nn.Module):
    """
    Soft Intersection over Union (IoU) Loss for binary segmentation.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_log: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred_log)
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        loss = (intersection_sum + self.smooth) / (pred_sum + target_sum - intersection_sum + self.smooth)
        return 1.0 - loss.mean()


class SLSIoULoss(nn.Module):
    """
    Scale-and-Location Sensitive Loss (SLSIoULoss).
    
    Combines scale-sensitive IoU penalty, location/angle distance regularization (LLoss),
    and warm-up scheduling for infrared small target segmentation.
    """
    def __init__(self, smooth: float = 0.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_log: torch.Tensor, target: torch.Tensor, warm_epoch: int = 5, epoch: int = 0, with_shape: bool = True) -> torch.Tensor:
        pred = torch.sigmoid(pred_log)
        smooth = self.smooth

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        dis = torch.pow((pred_sum - target_sum) / 2.0, 2)
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth)

        loss = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)
        lloss = compute_location_loss(pred, target)

        if epoch > warm_epoch:
            siou_loss = alpha * loss
            if with_shape:
                loss = 1.0 - siou_loss.mean() + lloss
            else:
                loss = 1.0 - siou_loss.mean()
        else:
            loss = 1.0 - loss.mean()

        return loss


def compute_location_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Location and Angle Distance Loss (LLoss).
    Computes spatial center displacement and directional angle discrepancy between prediction and ground truth.
    """
    loss = torch.tensor(0.0, requires_grad=True).to(pred)
    batch_size = pred.shape[0]
    h, w = pred.shape[2], pred.shape[3]

    x_index = torch.arange(0, w, 1, dtype=torch.float32, device=pred.device).view(1, 1, w).repeat(1, h, 1) / w
    y_index = torch.arange(0, h, 1, dtype=torch.float32, device=pred.device).view(1, h, 1).repeat(1, 1, w) / h
    eps = 1e-8

    for i in range(batch_size):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()

        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        angle_loss = (4.0 / (math.pi ** 2)) * torch.square(
            torch.arctan(pred_centery / (pred_centerx + eps)) - torch.arctan(target_centery / (target_centerx + eps))
        )

        pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + eps)
        target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + eps)

        length_loss = torch.min(pred_length, target_length) / (torch.max(pred_length, target_length) + eps)
        loss = loss + (1.0 - length_loss + angle_loss) / batch_size

    return loss


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count