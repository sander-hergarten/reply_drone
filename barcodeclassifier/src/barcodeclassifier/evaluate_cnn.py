"""
evaluate_cnn.py
---------------
Loads the trained TinyCNN and prints:
    • Accuracy
    • Precision, Recall, F1 at threshold 0.5
    • ROC-AUC

It re-uses the synthetic patch folders:
    data/patches/pos   (label = 1)
    data/patches/neg   (label = 0)

Run:
    rye run python -m barcodeclassifier.evaluate_cnn
"""

from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
from PIL import Image

from barcodeclassifier.models.cnn import TinyCNN


# -------- helper to load the val set -----------------
def build_loader(root="data/patches", batch=512):
    tf = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
    X, y = [], []
    for cls, sub in [(1, "pos"), (0, "neg")]:
        for p in (Path(root) / sub).glob("*.png"):
            X.append(tf(Image.open(p).convert("RGB")))
            y.append(cls)
    ds = TensorDataset(torch.stack(X), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch, shuffle=False)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCNN.load("models/tinycnn_labels.pt", map_location=device)
    loader = build_loader()

    probs, labels = [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Evaluating"):
            p = model(X.to(device)).cpu()
            probs.append(p); labels.append(y)
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()

    # metrics
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    auc = roc_auc_score(labels, probs)

    print(f"Accuracy  : {acc:6.3f}")
    print(f"Precision : {prec:6.3f}")
    print(f"Recall    : {rec:6.3f}")
    print(f"F1-score  : {f1:6.3f}")
    print(f"ROC-AUC   : {auc:6.3f}")


if __name__ == "__main__":
    main()
