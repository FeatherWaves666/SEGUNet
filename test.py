import os
import os.path as osp
import torch
import torch.utils.data as Data
from argparse import ArgumentParser
from tqdm import tqdm

from model.SEG_UNet import SEG_UNet
from utils.data import IRSTD_Dataset
from utils.metric import ROCMetric, PD_FA, mIoU, nIoU


def parse_args():
    parser = ArgumentParser(description='SEG-UNet: Evaluation & Testing Script')

    parser.add_argument('--dataset-dir', type=str, default='./dataset/IRSTD-1k', help='Path to IRSTD test dataset')
    parser.add_argument('--weight-path', type=str, required=True, help='Path to model checkpoint (.pth or .tar)')
    parser.add_argument('--base-size', type=int, default=256, help='Base image resolution (default: 256)')
    parser.add_argument('--crop-size', type=int, default=256, help='Crop image resolution (default: 256)')
    parser.add_argument('--save-dir', type=str, default='./results', help='Directory to save test outputs')

    return parser.parse_args()


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on device: {device}")

    valset = IRSTD_Dataset(args, mode='val')
    val_loader = Data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

    model = SEG_UNet(input_channels=1).to(device)

    if not osp.exists(args.weight_path):
        raise FileNotFoundError(f"Weight file not found: {args.weight_path}")

    print(f"Loading weights from {args.weight_path}...")
    ckpt = torch.load(args.weight_path, map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    eval_mIoU = mIoU(1)
    eval_nIoU = nIoU()
    eval_PD_FA = PD_FA(1, 10, args.base_size)
    eval_ROC = ROCMetric(1, 10)

    tbar = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad():
        for i, (data, mask) in enumerate(tbar):
            data = data.to(device)
            mask = mask.to(device)

            # Single-scale forward pass during inference
            _, pred = model(data, warm_flag=False)

            eval_mIoU.update(pred, mask)
            eval_nIoU.update(pred, mask)
            eval_PD_FA.update(pred, mask)
            eval_ROC.update(pred, mask)

            pixAcc, current_iou = eval_mIoU.get()
            current_niou = eval_nIoU.get()
            tbar.set_description(f"Eval | IoU: {current_iou * 100:.2f}% | nIoU: {float(current_niou) * 100:.2f}%")

    FA, PD = eval_PD_FA.get(len(val_loader))
    pixAcc, final_iou = eval_mIoU.get()
    final_niou = float(eval_nIoU.get())

    print("\n==============================================")
    print("         SEG-UNet Benchmark Evaluation        ")
    print("==============================================")
    print(f"  Intersection over Union (IoU) : {final_iou * 100:.2f}%")
    print(f"  Normalized IoU (nIoU)          : {final_niou * 100:.2f}%")
    print(f"  Probability of Detection (Pd)  : {PD[0] * 100:.2f}%")
    print(f"  False Alarm Rate (Fa)          : {FA[0] * 1e6:.2f} (10^-6)")
    print("==============================================\n")

    os.makedirs(args.save_dir, exist_ok=True)
    out_txt = osp.join(args.save_dir, 'evaluation_results.txt')
    with open(out_txt, 'a') as f:
        f.write(f"Weights: {args.weight_path}\n")
        f.write(f"IoU: {final_iou * 100:.2f}% | nIoU: {final_niou * 100:.2f}% | Pd: {PD[0] * 100:.2f}% | Fa: {FA[0] * 1e6:.2f}\n\n")

    print(f"Results saved to {out_txt}")


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
