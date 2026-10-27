#!/usr/bin/env python3
"""Train the working / not-working classifier on seated people. Runs on the GPU laptop.

    python3 tools/train_activity.py recordings/activity --res full --cv    # honest 4-fold test
    python3 tools/train_activity.py recordings/activity --res 720 --cv     # same, at the live 1280x720 resolution
    python3 tools/train_activity.py recordings/activity --res 720 --cv --by-day   # leave one whole day out
    python3 tools/train_activity.py recordings/activity --res 720          # train, save models/activity_d20.pt
    python3 tools/train_activity.py recordings/activity --res 720 --without 20261026-log --as d20_cand   # tools/weekly.sh

Labels come from recordings/activity/labels.json (tools/label_activity.py): W working, P phone,
T talking, I idle -> not working; U and unanswered are left out. Crops are cut fresh from the
frames (without the labelling outline): the desk box plus MARGIN, from the full 2560x1440
frame (--res full) or from the frame first shrunk to 1280x720 as the live system sees it
(--res 720). Folds are whole half-hours, as for the seat model. TorchVision
MobileNetV3-Large (BSD), fine-tuned from ImageNet weights.
"""
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS)); sys.path.insert(0, str(WS / "tools"))
MARGIN, SIZE, MAX_SIDE = 0.15, 320, 384
EPOCHS, BATCH, LR = 15, 32, 5e-4
NOT_WORKING = {"P", "T", "I"}


def examples(manifest, labels):
    out = []
    for m in manifest:
        a = labels.get(m["file"])
        if a == "W" or a in NOT_WORKING:
            out.append({**m, "y": int(a != "W"), "why": a})
    return out


def main():
    import cv2
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
    from crops import crop_box
    from presence import load_boxes
    from train_posture import letterbox_size
    from train_seat import fold_key, option

    root = Path(sys.argv[1])
    res = sys.argv[sys.argv.index("--res") + 1] if "--res" in sys.argv else "720"
    cv = "--cv" in sys.argv
    by_day = "--by-day" in sys.argv
    args = sys.argv[1:]
    name, without = option(args, "--as", "d20"), option(args, "--without", None)
    ex = [e for e in examples(json.loads((root / "manifest.json").read_text()), json.loads((root / "labels.json").read_text()))
          if e["dir"] != without]
    cache = root / f"train_{res}"
    cache.mkdir(exist_ok=True)
    for t_dir in sorted({e["dir"] for e in ex}):
        day = WS / "recordings" / t_dir
        boxes = load_boxes(day / "d20_focus.json")
        for e in [e for e in ex if e["dir"] == t_dir]:
            f = cache / e["file"]
            if f.exists():
                continue
            img = cv2.imread(str(day / "frames" / f"d20_{e['t']:05d}.jpg"))
            b = boxes[e["box"]]
            if res == "720":
                img, b = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA), tuple(v / 2 for v in b)
            x1, y1, x2, y2 = crop_box(b, img.shape[1], img.shape[0], MARGIN)
            c = img[y1:y2, x1:x2]
            s = MAX_SIDE / max(c.shape[:2])
            cv2.imwrite(str(f), cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else c,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"res {res}: {len(ex)} examples, working {sum(1 - e['y'] for e in ex)}, not working {sum(e['y'] for e in ex)}", flush=True)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def letterbox(img):
        w, h = letterbox_size(*img.size, SIZE)
        canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
        canvas.paste(img.resize((w, h)), ((SIZE - w) // 2, (SIZE - h) // 2))
        return canvas

    aug = transforms.Compose([transforms.ColorJitter(0.3, 0.3, 0.2),
                              transforms.RandomAffine(4, translate=(0.04, 0.04), scale=(0.93, 1.07), fill=114)])
    to_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    class Crops(Dataset):
        def __init__(self, xs, train):
            self.xs, self.train = xs, train

        def __len__(self):
            return len(self.xs)

        def __getitem__(self, i):
            img = letterbox(Image.open(cache / self.xs[i]["file"]).convert("RGB"))
            return to_t(aug(img) if self.train else img), self.xs[i]["y"]

    def fit(train, val):
        torch.manual_seed(0)
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
        model.cuda()
        pos = sum(e["y"] for e in train)
        w = torch.tensor([len(train) / (2 * (len(train) - pos)), len(train) / (2 * pos)], dtype=torch.float)
        loss_fn = nn.CrossEntropyLoss(weight=w.cuda())
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        tl = DataLoader(Crops(train, True), batch_size=BATCH, shuffle=True, num_workers=6)
        for _ in range(EPOCHS):
            model.train()
            for x, y in tl:
                opt.zero_grad()
                loss_fn(model(x.cuda()), y.cuda()).backward()
                opt.step()
            sched.step()
        model.eval()
        probs = []
        if val:
            with torch.no_grad():
                for x, _ in DataLoader(Crops(val, False), batch_size=64, num_workers=6):
                    out = model(x.cuda())
                    probs += zip(out.softmax(1)[:, 1].tolist(), (out[:, 1] - out[:, 0]).tolist())   # (p, raw logit)
        return probs, model

    if cv:
        keys = sorted({fold_key(e, by_day) for e in ex})
        assert len(keys) > 1, "--by-day needs at least two labelled days"
        rows = []
        for f in keys:
            train = [e for e in ex if fold_key(e, by_day) != f]
            val = [e for e in ex if fold_key(e, by_day) == f]
            probs, _ = fit(train, val)
            rows += [{**e, "p_not": round(p, 4), "logit": round(z, 4)} for e, (p, z) in zip(val, probs)]
            print(f"fold {f}: trained on {len(train)}, tested on {len(val)}", flush=True)
        pred = lambda r: r["p_not"] >= 0.5
        work = [r for r in rows if r["y"] == 0]; notw = [r for r in rows if r["y"] == 1]
        wr = sum(not pred(r) for r in work) / len(work); nr = sum(pred(r) for r in notw) / len(notw)
        acc = sum(pred(r) == bool(r["y"]) for r in rows) / len(rows)
        print(f"\n=== res {res}: {'leave-one-day-out' if by_day else '4-fold CV by half-hour'}, {len(rows)} seated people ===")
        print(f"overall {acc:.1%} | working recognised {wr:.1%} ({len(work)}) | not working caught {nr:.1%} ({len(notw)}) | balanced {(wr + nr) / 2:.1%}")
        for why in ("P", "T", "I"):
            g = [r for r in rows if r["why"] == why]
            print(f"   {why}: caught {sum(pred(r) for r in g)}/{len(g)}")
        for b in sorted({r["box"] for r in rows}):
            g = [r for r in rows if r["box"] == b]
            print(f"   desk {b}: right {sum(pred(r) == bool(r['y']) for r in g)}/{len(g)}  (not working {sum(r['y'] for r in g)})")
        (WS / "models" / f"activity_d20_cv_{res}{'_byday' if by_day else ''}.json").write_text(json.dumps(rows))
    else:
        _, model = fit(ex, [])
        torch.save(model.state_dict(), WS / "models" / f"activity_{name}.pt")
        (WS / "models" / f"activity_{name}.json").write_text(json.dumps(
            {"res": res, "margin": MARGIN, "size": SIZE, "max_side": MAX_SIDE, "mean": mean, "std": std,
             "letterbox_fill": 114, "classes": ["working", "not working"], "arch": "mobilenet_v3_large",
             "examples": len(ex), "trained_on": sorted({e["dir"] for e in ex})}, indent=1))
        print(f"saved models/activity_{name}.pt")


if __name__ == "__main__":
    main()
