"""Seat-occupied classifier at runtime: one picture per desk box -> probability someone sits there.

Same preparation as training/train_seat.py: the box plus its margin is cut out, shrunk to at
most 320 px, padded to a square (not stretched) and normalised like ImageNet. Box coordinates
are scaled with the frame, so the cut-out covers the same part of the room as in training.
"""
import json

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from seatsense.paths import ROOT
from seatsense.crops import crop_box, letterbox_size


def available(cam):
    return (ROOT / "models" / f"seat_{cam}.pt").exists()


class SeatModel:
    def __init__(self, cam):
        meta = json.loads((ROOT / "models" / f"seat_{cam}.json").read_text())
        self.margin, self.size, self.fill = meta["margin"], meta["size"], meta["letterbox_fill"]
        m = mobilenet_v3_small()
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
        m.load_state_dict(torch.load(ROOT / "models" / f"seat_{cam}.pt", map_location="cpu", weights_only=True))
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
