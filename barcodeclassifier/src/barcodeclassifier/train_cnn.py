"""
train_tinycnn.py
----------------
1) Generates synthetic patches if they don't exist.
2) Trains TinyCNN for 5 epochs on CPU.
3) Saves model to models/tinycnn_labels.pt

Run:
    rye run python -m cratescout_vision.train_tinycnn
"""

from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from tqdm import tqdm
import PIL
from barcodeclassifier.models.cnn import TinyCNN
from barcodeclassifier.utils.generate_synth_patches import main as gen_patches

# ---------- dataset ---------- #
class PatchFolder(Dataset):
    def __init__(self, root: Path, label: int):
        self.paths = list(root.glob("*.png"))
        tf = Compose([ToTensor(), Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        self.samples = [(tf(PIL.Image.open(p).convert("RGB")), label) for p in self.paths]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def build_loader(root="data/patches", batch=256) -> DataLoader:
    print("🔄 Loading synthetic patches")
    root = Path(root)
    if not root.exists():
        print("Synthetic patches not found – generating …")
        gen_patches(12_000, 20_000)          # default amounts
    pos_ds = PatchFolder(root/"pos", 1)
    neg_ds = PatchFolder(root/"neg", 0)
    ds = pos_ds.samples + neg_ds.samples
    X, Y = zip(*ds)
    tensor_ds = torch.utils.data.TensorDataset(torch.stack(X), torch.tensor(Y, dtype=torch.float32))
    return DataLoader(tensor_ds, batch_size=batch, shuffle=True, num_workers=2)

# ---------- training ---------- #
def train():
    print("🔄 Training TinyCNN on synthetic patches")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_loader()
    model = TinyCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCELoss()

    for epoch in range(5):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/5")
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            p = model(X)
            loss = crit(p, y)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss))

    Path("models").mkdir(exist_ok=True)
    model.save("models/tinycnn_labels.pt")
    print("✅ Model saved to models/tinycnn_labels.pt")

if __name__ == "__main__":
    from PIL import Image            # import here to satisfy lazy loader
    train()
