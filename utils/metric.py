import numpy as np
import torch
import torch.nn as nn
from skimage import measure

class ROCMetric:
    """
    Computes ROC metrics (True Positive Rate, False Positive Rate, Recall, Precision) across thresholds.
    """
    def __init__(self, nclass: int = 1, bins: int = 10):
        self.nclass = nclass
        self.bins = bins
        self.reset()

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) / self.bins
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 1e-3)
        fp_rates = self.fp_arr / (self.neg_arr + 1e-3)
        recall = self.tp_arr / (self.pos_arr + 1e-3)
        precision = self.tp_arr / (self.class_pos + 1e-3)
        return tp_rates, fp_rates, recall, precision

    def reset(self):
        self.tp_arr = np.zeros([self.bins + 1])
        self.pos_arr = np.zeros([self.bins + 1])
        self.fp_arr = np.zeros([self.bins + 1])
        self.neg_arr = np.zeros([self.bins + 1])
        self.class_pos = np.zeros([self.bins + 1])


class PD_FA:
    """
    Computes Probability of Detection (Pd) and False Alarm Rate (Fa).
    Target centroid distance threshold < 3 pixels for true detection.
    """
    def __init__(self, nclass: int = 1, bins: int = 10, size: int = 256):
        self.nclass = nclass
        self.bins = bins
        self.size = size
        self.reset()

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        for iBin in range(self.bins + 1):
            score_thresh = iBin * (255.0 / self.bins)
            predits = np.array((preds > score_thresh).cpu()).astype('int64')
            predits = np.reshape(predits, (self.size, self.size))

            labelss = np.array(labels.cpu()).astype('int64')
            labelss = np.reshape(labelss, (self.size, self.size))

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss, connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin] += len(coord_label)
            image_area_total = []
            image_area_match = []
            distance_match = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        distance_match.append(distance)
                        image_area_match.append(area_image)
                        del coord_image[m]
                        break

            dismatch = [x for x in image_area_total if x not in image_area_match]
            self.FA[iBin] += np.sum(dismatch)
            self.PD[iBin] += len(distance_match)

    def get(self, img_num: int):
        Final_FA = self.FA / ((self.size * self.size) * img_num)
        Final_PD = self.PD / (self.target + 1e-6)
        return Final_FA, Final_PD

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])
        self.target = np.zeros([self.bins + 1])


class mIoU:
    """
    Computes Pixel Accuracy (pixAcc) and Mean Intersection over Union (mIoU).
    """
    def __init__(self, nclass: int = 1):
        self.nclass = nclass
        self.reset()

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        return pixAcc, IoU.mean()

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


class nIoU:
    """
    Computes Normalized IoU (nIoU) metric.
    nIoU = (1/N) * sum(TP[i] / (T[i] + P[i] - TP[i]))
    """
    def __init__(self):
        self.reset()

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        predict = (preds > 0).float()
        if len(labels.shape) == 3:
            target = labels.float().unsqueeze(1)
        elif len(labels.shape) == 4:
            target = labels.float()
        else:
            raise ValueError("Unknown target dimension")

        assert predict.shape == target.shape, f"Shape mismatch: {predict.shape} vs {target.shape}"
        true_positive = (predict * target).sum()
        true_total = target.sum()
        pred_total = predict.sum()

        self.sum_niou += true_positive / (true_total + pred_total - true_positive + np.spacing(1))
        self.num_images += 1

    def get(self):
        return self.sum_niou / (self.num_images + np.spacing(1))

    def reset(self):
        self.sum_niou = 0.0
        self.num_images = 0


def cal_tp_pos_fp_neg(output: torch.Tensor, target: torch.Tensor, nclass: int, score_thresh: float):
    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = target.float().unsqueeze(1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())
    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos = tp + fp
    return tp, pos, fp, neg, class_pos


def batch_pix_accuracy(output: torch.Tensor, target: torch.Tensor):
    if len(target.shape) == 3:
        target = target.float().unsqueeze(1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, f"Shape mismatch: {output.shape} vs {target.shape}"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float()) * ((target > 0).float())).sum()
    assert pixel_correct <= pixel_labeled, "Correct pixel area should be <= labeled area"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output: torch.Tensor, target: torch.Tensor, nclass: int):
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = target.float().unsqueeze(1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())
    area_inter, _ = np.histogram(intersection.cpu(), bins=1, range=(1, 1))
    area_pred, _ = np.histogram(predict.cpu(), bins=1, range=(1, 1))
    area_lab, _ = np.histogram(target.cpu(), bins=1, range=(1, 1))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be <= union area"
    return area_inter, area_union
