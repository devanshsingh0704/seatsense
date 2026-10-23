"""Seat-occupied classifier at runtime: one picture per desk box -> probability someone sits there.

Same preparation as tools/train_seat.py: the box plus its margin is cut out, shrunk to at
most 320 px, padded to a square (not stretched) and normalised like ImageNet. Box coordinates
are scaled with the frame, so the cut-out covers the same part of the room as in training.
"""
import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "tools"))
from crops import crop_box                 # noqa: E402
from train_posture import letterbox_size   # noqa: E402


def available(cam):
    return (HERE / "models" / f"seat_{cam}.pt").exists()


class SeatModel:
    def __init__(self, cam):
        meta = json.loads((HERE / "models" / f"seat_{cam}.json").read_text())
        self.margin, self.size, self.fill = meta["margin"], meta["size"], meta["letterbox_fill"]
        m = mobilenet_v3_small()
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
        m.load_state_dict(torch.load(HERE / "models" / f"seat_{cam}.pt", map_location="cpu"))
        self.model = m.eval().cuda()
        self.to_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(meta["mean"], meta["std"])])

    def _letterbox(self, img):
        w, h = letterbox_size(*img.size, self.size)
        canvas = Image.new("RGB", (self.size, self.size), (self.fill,) * 3)
        canvas.paste(img.resize((w, h)), ((self.size - w) // 2, (self.size - h) // 2))
        return canvas

    @torch.no_grad()
    def predict(self, frame, boxes):
        """frame: BGR image; boxes: {n: (x1, y1, x2, y2)} in that frame's pixels -> {n: p_sitting}."""
        h, w = frame.shape[:2]
        batch = []
        for b in boxes.values():
            x1, y1, x2, y2 = crop_box(b, w, h, self.margin)
            c = frame[y1:y2, x1:x2]
            s = 320 / max(c.shape[:2])
            if s < 1:
                c = cv2.resize(c, None, fx=s, fy=s)
            batch.append(self.to_t(self._letterbox(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))))
        p = self.model(torch.stack(batch).cuda()).softmax(1)[:, 1].tolist()
        return dict(zip(boxes, p))
