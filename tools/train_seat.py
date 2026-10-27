#!/usr/bin/env python3
"""Train the "seat occupied?" classifier: one picture per desk box -> sitting / empty.

    python3 tools/train_seat.py recordings/20261012-day [recordings/20261017-log ...] --cv   # honest 4-fold test
    python3 tools/train_seat.py recordings/20261012-day recordings/20261017-log --cv --by-day   # train on the other days, test on a new one
    python3 tools/train_seat.py recordings/20261012-day [recordings/20261017-log ...]        # train, save models/seat_d20.pt
    python3 tools/train_seat.py ... --as d20_cand        # save as models/seat_d20_cand.pt (tools/weekly.sh: a candidate)

Several folders: each needs labels_d20.json, d20_focus.json and frames/; only labelled frames are used.

Each labelled frame gives one example per box: the box region (plus MARGIN) cut from the
full-resolution frame, labelled sitting if the labeller marked that box. One model is shared
by all boxes. Folds are whole HALF-HOURS of the day (minute-apart frames are near-identical,
so a random split would test on pictures the model has effectively seen). The baseline is
the current rule's answer on the same frames (prelabels_d20.json).
PyTorch + torchvision MobileNetV3-Small (BSD), fine-tuned from ImageNet weights.
"""
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS / "tools"))
CAM = "d20"
MARGIN = 0.1
SIZE = 224
BLOCK_S, FOLDS = 1800, 4
EPOCHS, BATCH, LR = 15, 64, 1e-3


def option(args, flag, default):
    """Remove `flag VALUE` from args (in place) and return VALUE, or default when absent."""
    if flag not in args:
        return default
    i = args.index(flag)
    value = args[i + 1]
    del args[i:i + 2]
    return value


def build_examples(labels, boxes):
    return [{"t": int(t), "box": n, "sitting": int(n in sat)} for t, sat in labels.items() for n in boxes]


def fold_of(t):
    return (t // BLOCK_S) % FOLDS


def fold_key(e, by_day):
    """--by-day holds out one whole day at a time (a new day, its light and its people's habits);
    otherwise half-hour blocks, which share days between training and test."""
    return e["dir"] if by_day else fold_of(e["t"])


def main():
    import cv2
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    from crops import crop_box, letterbox_size
    from presence import load_boxes

    args = sys.argv[1:]
    name = option(args, "--as", CAM)
    days = [Path(a) for a in args if not a.startswith("--")]
    cv = "--cv" in sys.argv
    by_day = "--by-day" in sys.argv
    boxes = load_boxes(days[0] / f"{CAM}_focus.json")
    ex, pre = [], {}
    for day in days:
        assert load_boxes(day / f"{CAM}_focus.json") == boxes, f"{day}: different desk boxes"
        labels = json.loads((day / f"labels_{CAM}.json").read_text())
        p = day / f"prelabels_{CAM}.json"
        for t, v in (json.loads(p.read_text()) if p.exists() else {}).items():
            pre[(str(day), int(t))] = v
        ex += [{**e, "dir": str(day)} for e in build_examples(labels, list(boxes))]
        cache = day / "seat_crops"        # crop once from the 2560x1440 frames, reuse every epoch and fold
        cache.mkdir(exist_ok=True)
        for t in sorted({int(t) for t in labels}):
            if all((cache / f"{CAM}_{t:05d}_{n}.jpg").exists() for n in boxes):
                continue
            img = cv2.imread(str(day / "frames" / f"{CAM}_{t:05d}.jpg"))
            for n, b in boxes.items():
                x1, y1, x2, y2 = crop_box(b, img.shape[1], img.shape[0], MARGIN)
                c = img[y1:y2, x1:x2]
                s = 320 / max(c.shape[:2])
                cv2.imwrite(str(cache / f"{CAM}_{t:05d}_{n}.jpg"), cv2.resize(c, None, fx=s, fy=s) if s < 1 else c,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{day.name}: {len(labels)} labelled frames", flush=True)
    print(f"{len(ex)} examples ({sum(e['sitting'] for e in ex)} sitting), boxes {list(boxes)}", flush=True)

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
            e = self.xs[i]
            img = letterbox(Image.open(Path(e["dir"]) / "seat_crops" / f"{CAM}_{e['t']:05d}_{e['box']}.jpg").convert("RGB"))
            return to_t(aug(img) if self.train else img), e["sitting"]

    def fit(train, val):
        torch.manual_seed(0)
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
        model.cuda()
        pos = sum(e["sitting"] for e in train)
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
        preds, probs, logits = [], [], []
        if val:
            with torch.no_grad():
                for x, _ in DataLoader(Crops(val, False), batch_size=128, num_workers=6):
                    out = model(x.cuda())
                    pr = out.softmax(1)[:, 1]
                    probs += pr.tolist()
                    logits += (out[:, 1] - out[:, 0]).tolist()   # raw score, for exact calibration later
                    preds += (pr >= 0.5).long().tolist()
        return preds, (probs, logits), model

    def report(name, rows):
        n = len(rows)
        right = sum(p == e["sitting"] for e, p in rows)
        tp = sum(p and e["sitting"] for e, p in rows); fn = sum((not p) and e["sitting"] for e, p in rows)
        fp = sum(p and not e["sitting"] for e, p in rows)
        print(f"{name:30s} {right}/{n} = {right / n:6.1%} | missed sitters {fn} | wrong 'sitting' {fp} | "
              f"sitters caught {tp}/{tp + fn} = {tp / max(1, tp + fn):.1%}", flush=True)
        for b in boxes:
            r = [(e, p) for e, p in rows if e["box"] == b]
            bfn = sum((not p) and e["sitting"] for e, p in r); bfp = sum(p and not e["sitting"] for e, p in r)
            print(f"     box {b}: {sum(p == e['sitting'] for e, p in r)}/{len(r)} right, missed {bfn}, wrong 'sitting' {bfp}")
        return {"accuracy": round(right / n, 4), "missed_sitters": fn, "wrong_sitting": fp}

    baseline = [(e, int(e["box"] in pre.get((e["dir"], e["t"]), []))) for e in ex]
    if cv:
        keys = sorted({fold_key(e, by_day) for e in ex})
        assert len(keys) > 1, "--by-day needs at least two labelled days"
        rows, dump = [], []
        for f in keys:
            train = [e for e in ex if fold_key(e, by_day) != f]
            val = [e for e in ex if fold_key(e, by_day) == f]
            preds, (probs, logits), _ = fit(train, val)
            rows += list(zip(val, preds))
            dump += [{**e, "p_sitting": round(pr, 4), "logit": round(z, 4), "rule": int(e["box"] in pre.get((e["dir"], e["t"]), []))}
                     for e, pr, z in zip(val, probs, logits)]
            print(f"fold {f}: trained on {len(train)}, tested on {len(val)}", flush=True)
            if by_day:
                report(f"  held-out day {Path(f).name}", list(zip(val, preds)))
        how = "leave-one-day-out" if by_day else "4-fold cross-validation by half-hour blocks"
        print(f"\n=== {how}, {len(rows)} desk checks ===")
        res = {"classifier": report("seat classifier", rows), "box_rule": report("baseline (prelabels)", baseline)}
        tag = "_byday" if by_day else ""
        (WS / "models" / f"seat_{CAM}_cv{tag}.json").write_text(json.dumps(res, indent=1))
        (WS / "models" / f"seat_{CAM}_cv{tag}_rows.json").write_text(json.dumps(dump))   # every held-out answer, for analysis
    else:
        _, _, model = fit(ex, [])
        torch.save(model.state_dict(), WS / "models" / f"seat_{name}.pt")
        (WS / "models" / f"seat_{name}.json").write_text(json.dumps(
            {"camera": CAM, "boxes": {n: b for n, b in boxes.items()}, "margin": MARGIN, "size": SIZE,
             "mean": mean, "std": std, "letterbox_fill": 114, "classes": ["empty", "sitting"],
             "trained_on": [str(d) for d in days], "examples": len(ex), "epochs": EPOCHS}, indent=1))
        print(f"saved models/seat_{name}.pt")


if __name__ == "__main__":
    main()
