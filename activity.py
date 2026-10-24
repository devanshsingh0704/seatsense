"""Working / not-working for a seated person (beta): the classifier from tools/train_activity.py
plus a hold, so a glance at a phone never counts.

"Not working" after 20 s of not-working looks (the user's rule; a misread of up to 3 s does not
restart the count), back to "working" after 10 s of working looks. The hold is presence.Desk
with "looks not working" in place of "someone is there". Preparation matches training at
--res 720: the desk box plus 15% from the 1280x720 frame, shrunk to at most 384 px, padded to
a 320 square with grey 114, normalised like ImageNet.
"""
import json
import sys
from pathlib import Path

from presence import Desk

HERE = Path(__file__).parent


def available(cam):
    return (HERE / "models" / f"activity_{cam}.pt").exists()


NOT_AT = 0.7   # flag "not working" from 0.7, not 0.5 (24 Oct, unseen days): wrong flags 144 -> 109, caught 67% -> 63%


class Activity:
    def __init__(self, not_after_s=20, back_after_s=10, gap_s=3):
        self.args = (not_after_s, back_after_s, gap_s)
        self.reset()

    def reset(self):
        self.hold = Desk(*self.args)

    def update(self, p_not, now):
        self.hold.update(p_not >= NOT_AT, now)
        return "not" if self.hold.state == "occupied" else "working"


class ActivityModel:
    def __init__(self, cam):
        import torch
        from torch import nn
        from torchvision import transforms
        from torchvision.models import mobilenet_v3_large
        sys.path.insert(0, str(HERE / "tools"))
        from crops import crop_box
        from train_posture import letterbox_size
        self.crop_box, self.letterbox_size = crop_box, letterbox_size
        meta = json.loads((HERE / "models" / f"activity_{cam}.json").read_text())
        self.margin, self.size, self.max_side, self.fill = meta["margin"], meta["size"], meta["max_side"], meta["letterbox_fill"]
        m = mobilenet_v3_large()
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
        m.load_state_dict(torch.load(HERE / "models" / f"activity_{cam}.pt", map_location="cpu"))
        self.model = m.eval().cuda()
        self.torch = torch
        self.to_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(meta["mean"], meta["std"])])

    def predict(self, frame, boxes):
        """frame: 1280x720 BGR image; boxes: {n: (x1, y1, x2, y2)} in its pixels -> {n: p_not_working}."""
        import cv2
        from PIL import Image
        if not boxes:
            return {}
        h, w = frame.shape[:2]
        batch = []
        for b in boxes.values():
            x1, y1, x2, y2 = self.crop_box(b, w, h, self.margin)
            c = frame[y1:y2, x1:x2]
            s = self.max_side / max(c.shape[:2])
            if s < 1:
                c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            img = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            lw, lh = self.letterbox_size(*img.size, self.size)
            canvas = Image.new("RGB", (self.size, self.size), (self.fill,) * 3)
            canvas.paste(img.resize((lw, lh)), ((self.size - lw) // 2, (self.size - lh) // 2))
            batch.append(self.to_t(canvas))
        with self.torch.no_grad():
            p = self.model(self.torch.stack(batch).cuda()).softmax(1)[:, 1].tolist()
        return dict(zip(boxes, p))
