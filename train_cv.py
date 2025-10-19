# file: train_cv.py
# Transfer-learn a ResNet50 classifier on product thumbnails. Produces a multi-class model saved as resnet50_prod.pth
# Usage: python train_cv.py
import os, json, random
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import shutil

DATA_DIR = Path("data_prep_output/images")
META = json.load(open("data_prep_output/meta.json",'r',encoding='utf-8'))
# map uniq_id -> first category (fallback 'unknown')
# read products_clean.jsonl for categories
prods = [json.loads(l) for l in open("data_prep_output/products_clean.jsonl",'r',encoding='utf-8').read().splitlines()]
id2cat = {p['uniq_id']:(p.get('categories') or ['unknown'])[0] for p in prods}

# create dataset dir with top classes
out_ds = Path("data_prep_output/cv_dataset")
if out_ds.exists():
    shutil.rmtree(out_ds)
(out_ds/"train").mkdir(parents=True, exist_ok=True)
(out_ds/"val").mkdir(parents=True, exist_ok=True)

# collect images and group by class
class_map = {}
for img in DATA_DIR.glob("*.jpg"):
    uid = "_".join(img.name.split("_")[:-1])
    cat = id2cat.get(uid, "unknown")
    class_map.setdefault(cat, []).append(img)

# pick top N classes with >=5 images
classes = [ (c, lst) for c,lst in class_map.items() if len(lst)>=3 ]
classes = sorted(classes, key=lambda x: len(x[1]), reverse=True)[:8]
for cat, imgs in classes:
    (out_ds/"train"/cat).mkdir(parents=True, exist_ok=True)
    (out_ds/"val"/cat).mkdir(parents=True, exist_ok=True)
    for i, imgp in enumerate(imgs):
        dst = out_ds/"train"/cat/f"{imgp.stem}.jpg" if i%8!=0 else out_ds/"val"/cat/f"{imgp.stem}.jpg"
        shutil.copy(imgp, dst)

# Dataloaders
train_tf = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
val_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

from torchvision.datasets import ImageFolder
train_ds = ImageFolder(out_ds/"train", transform=train_tf)
val_ds = ImageFolder(out_ds/"val", transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_ds.classes))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# tiny training loop
for epoch in range(3):
    model.train()
    total=0; acc=0
    for xb,yb in train_loader:
        xb,yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += yb.size(0)
        acc += (pred.argmax(1)==yb).sum().item()
    print(f"Epoch {epoch} train acc {acc/total:.3f}")
    # val
    model.eval()
    total=0; acc=0
    with torch.no_grad():
        for xb,yb in val_loader:
            xb,yb=xb.to(device), yb.to(device)
            pred = model(xb)
            total += yb.size(0)
            acc += (pred.argmax(1)==yb).sum().item()
    print(f"Epoch {epoch} val acc {acc/total:.3f}")

torch.save({"model_state":model.state_dict(), "classes": train_ds.classes}, "data_prep_output/resnet50_prod.pth")
print("Saved resnet50_prod.pth")
