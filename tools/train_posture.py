#!/usr/bin/env python3
"""Train the sitting/standing classifier on labelled person crops. Runs on the GPU laptop.

    python3 tools/train_posture.py recordings/20261011-153440

PyTorch + torchvision MobileNetV3-Small (BSD), fine-tuned from ImageNet weights.
Validation holds out whole BLOCKS of time, never single crops: crops of the same seated
person 20 s apart are near-identical, and a random split would score the model on
pictures it has effectively already seen. Crops are padded to a square, not stretched,
because a person's shape is part of telling sitting from standing. The box rule's guess
on the same held-out crops is reported as the baseline to beat.
Writes models/posture_mnv3s.pt (weights) and models/posture_mnv3s.json (settings + scores).
"""
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
CLASSES = ["sit", "stand"]
SIZE = 224
BLOCK_S, EVERY = 120, 4          # every 4th 2-minute block of each camera is held out
EPOCHS, BATCH, LR = 25, 32, 1e-3


def split_by_time(items, block_s=BLOCK_S, every=EVERY, offset=0):
    train, val = [], []
    for it in items:
        (val if (it["t"] // block_s) % every == offset else train).append(it)
    return train, val


def letterbox_size(w, h, size=SIZE):
    s = size / max(w, h)
    return round(w * s), round(h * s)


def main():
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    rec = Path(sys.argv[1])
    cv = "--cv" in sys.argv     # 4-fold check: each time block held out once, final epoch only
    crops = rec / "crops"
    manifest = json.loads((crops / "manifest.json").read_text())
    labels = json.loads((crops / "labels.json").read_text())
    items = [{**m, "label": labels[m["file"]]} for m in manifest if labels.get(m["file"]) in CLASSES]

    count = lambda xs, c: sum(it["label"] == c for it in xs)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def letterbox(img):
        w, h = letterbox_size(*img.size)
        canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
        canvas.paste(img.resize((w, h)), ((SIZE - w) // 2, (SIZE - h) // 2))
        return canvas

    aug = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.3, 0.3, 0.2),
                              transforms.RandomAffine(5, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=114)])
    to_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    class Crops(Dataset):
        def __init__(self, xs, train):
            self.xs, self.train = xs, train

        def __len__(self):
            return len(self.xs)

        def __getitem__(self, i):
            img = letterbox(Image.open(crops / self.xs[i]["file"]).convert("RGB"))
            return to_t(aug(img) if self.train else img), CLASSES.index(self.xs[i]["label"])

    dev = "cuda"

    def fit(train, val):
        """Train from ImageNet weights; return (final-epoch predictions on val, final weights)."""
        torch.manual_seed(0)
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
        model.to(dev)
        freq = torch.tensor([count(train, c) for c in CLASSES], dtype=torch.float)
        loss_fn = nn.CrossEntropyLoss(weight=(freq.sum() / freq / len(CLASSES)).to(dev))   # standing is rare
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        tl = DataLoader(Crops(train, True), batch_size=BATCH, shuffle=True, num_workers=4)
        for _ in range(EPOCHS):
            model.train()
            for x, y in tl:
                opt.zero_grad()
                loss_fn(model(x.to(dev)), y.to(dev)).backward()
                opt.step()
            sched.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in DataLoader(Crops(val, False), batch_size=64, num_workers=4):
                preds += model(x.to(dev)).argmax(1).tolist()
        return preds, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def report(name, truth, pred):
        conf = [[0, 0], [0, 0]]
        for t, p in zip(truth, pred):
            conf[t][p] += 1
        n = len(truth)
        acc = (conf[0][0] + conf[1][1]) / n
        sit, stand = conf[0][0] / max(1, sum(conf[0])), conf[1][1] / max(1, sum(conf[1]))
        print(f"{name:34s} overall {acc:6.1%}  sitting right {sit:6.1%} ({conf[0][0]}/{sum(conf[0])})  "
              f"standing right {stand:6.1%} ({conf[1][1]}/{sum(conf[1])})", flush=True)
        return {"overall": round(acc, 3), "sitting": round(sit, 3), "standing": round(stand, 3), "confusion": conf}

    folds = range(EVERY) if cv else [0]
    truth, pred_clf, pred_box, pred_both = [], [], [], []
    for o in folds:
        train, val = split_by_time(items, offset=o)
        print(f"fold {o}: train {len(train)} (stand {count(train, 'stand')}) | val {len(val)} (stand {count(val, 'stand')})", flush=True)
        preds, state = fit(train, val)
        for it, p in zip(val, preds):
            truth.append(CLASSES.index(it["label"]))
            pred_clf.append(p)
            pred_box.append(CLASSES.index(it["guess"]))
            # combination: seated only if the classifier says sit AND at least half of them is in the box
            pred_both.append(0 if p == 0 and it["inside"] >= 0.5 else 1)

    print(f"\n=== {'4-fold cross-validation, every crop held out once' if cv else 'one held-out split'}: {len(truth)} crops ===")
    scores = {"classifier": report("MobileNetV3 classifier", truth, pred_clf),
              "box_rule": report("box rule (85% inside)", truth, pred_box),
              "classifier_and_half_in_box": report("classifier AND >=50% in box", truth, pred_both)}
    if not cv:   # the deployable model: trained on the main split
        torch.save(state, WS / "models" / "posture_mnv3s.pt")
    meta = {"classes": CLASSES, "size": SIZE, "mean": mean, "std": std, "letterbox_fill": 114,
            "trained_on": str(rec), "epochs": EPOCHS, "split": {"block_s": BLOCK_S, "every": EVERY},
            "evaluation": "4-fold time-block CV" if cv else "single split", "scores": scores}
    (WS / "models" / ("posture_cv.json" if cv else "posture_mnv3s.json")).write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
