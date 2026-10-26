"""
CNN document classifier (transfer learning) -> exported to ONNX for CPU inference.
Run on Kaggle / Colab with a free GPU (see docs/GUIDE_AR.md, phase 4).

    pip install torch torchvision onnx
    python scripts/train_classifier_torch.py --data data/synthetic --epochs 8

Data layout expected (synthetic_docs.py already produces it):
    data/synthetic/images/{train,val,test}/<class>/*.jpg
Add REAL public images to the same folders to fight the synthetic->real gap
(e.g. RVL-CDIP letters/invoices -> other/financial, MIDV-2020 IDs -> id_card/passport).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]


def loaders(data: Path, bs: int):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.6, 1.6)),
        transforms.RandomRotation(8), transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.RandomGrayscale(0.2), transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    ds = {s: datasets.ImageFolder(data / "images" / s, train_tf if s == "train" else eval_tf)
          for s in ("train", "val", "test")}
    dl = {s: DataLoader(d, batch_size=bs, shuffle=(s == "train"), num_workers=2) for s, d in ds.items()}
    return ds, dl


def evaluate(model, dl, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dl:
            p = model(x.to(device)).argmax(1).cpu()
            correct += (p == y).sum().item()
            total += len(y)
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, dl = loaders(Path(a.data), a.bs)
    classes = ds["train"].classes
    print("classes:", classes, "device:", device)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best, out = 0.0, ROOT / "models"
    out.mkdir(exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        for x, y in dl["train"]:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        sched.step()
        val = evaluate(model, dl["val"], device)
        print(f"epoch {ep + 1}: loss={loss.item():.3f} val_acc={val:.3f}")
        if val >= best:
            best = val
            torch.save(model.state_dict(), out / "doc_classifier.pt")

    model.load_state_dict(torch.load(out / "doc_classifier.pt", map_location=device))
    test = evaluate(model, dl["test"], device)
    print(f"TEST accuracy: {test:.3f}")

    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), out / "doc_classifier.onnx",
                      input_names=["image"], output_names=["logits"], dynamic_axes={"image": {0: "batch"}},
                      opset_version=17)
    (out / "doc_classifier.json").write_text(json.dumps({"classes": classes, "test_accuracy": test}))
    print("exported -> models/doc_classifier.onnx (copy it + doc_classifier.json into your repo's models/)")


if __name__ == "__main__":
    main()
