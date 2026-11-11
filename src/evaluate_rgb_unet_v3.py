from pathlib import Path
import torch
from torch.utils.data import DataLoader
from rgb_flood_dataset import RGBFloodDataset
from rgb_unet_model import RGBUNet

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
CHECKPOINT_PATH = PROJECT_DIR / "models" / "rgb_unet_v3_best.pth"
DEVICE = torch.device("cpu")
torch.set_num_threads(2)

def metrics(tp, fp, fn, tn):
    eps = 1e-8
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)
    return p, r, f1, iou, acc

def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    threshold = float(ckpt.get("validation_threshold", 0.30))
    model = RGBUNet(
        in_channels=int(ckpt.get("in_channels", 3)),
        base_channels=int(ckpt.get("base_channels", 8)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = RGBFloodDataset(split="test", augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    tp = fp = fn = tn = skipped = evaluated = 0

    with torch.no_grad():
        for i, (images, labels, valid) in enumerate(loader, 1):
            images = images.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE).float()
            valid = valid.unsqueeze(1).to(DEVICE).bool()
            if not valid.any().item():
                skipped += 1
                continue
            probs = torch.sigmoid(model(images))
            pred = probs >= threshold
            target = labels >= 0.5
            tp += (pred & target & valid).sum().item()
            fp += (pred & ~target & valid).sum().item()
            fn += (~pred & target & valid).sum().item()
            tn += (~pred & ~target & valid).sum().item()
            evaluated += 1
            if i % 10 == 0:
                print(f"Processed {i}/{len(loader)} test samples")

    p, r, f1, iou, acc = metrics(tp, fp, fn, tn)
    total = tp + fp + fn + tn
    print("\n" + "=" * 60)
    print("RGB U-NET V3 FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Best epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Stored validation F1: {float(ckpt.get('validation_f1', 0)):.4f}")
    print(f"Threshold selected on validation: {threshold:.2f}")
    print(f"Test samples: {len(dataset)} | evaluated: {evaluated} | skipped: {skipped}")
    print(f"Valid test pixels: {total:,}")
    print(f"TP: {int(tp):,} | FP: {int(fp):,} | FN: {int(fn):,} | TN: {int(tn):,}")
    print(f"Precision: {p:.4f} ({p*100:.2f}%)")
    print(f"Recall:    {r:.4f} ({r*100:.2f}%)")
    print(f"F1-score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"Water IoU: {iou:.4f} ({iou*100:.2f}%)")
    print(f"Pixel accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Predicted water pixels: {(tp+fp)/(total+1e-8)*100:.2f}%")
    print(f"Reference water pixels: {(tp+fn)/(total+1e-8)*100:.2f}%")
    print("\nKeep this threshold fixed for test evaluation; select thresholds using validation data.")

if __name__ == "__main__":
    main()
