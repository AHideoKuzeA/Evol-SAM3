import os
import datetime
import numpy as np
import cv2
import torch
import json
from io import BytesIO
from PIL import Image, ImageDraw

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0: return 0.0
    return np.sum(intersection) / np.sum(union)

def get_mask_from_json(json_path, img):
    try:
        with open(json_path, "r") as r: anno = json.loads(r.read())
    except:
        return np.zeros(img.shape[:2], dtype=np.uint8), "", False

    inform = anno.get("shapes", [])
    is_sentence = anno.get("is_sentence", False)
    comments = anno.get("text", "")
    height, width = img.shape[:2]
    area_list = []
    valid_poly_list = []
    for i in inform:
        if "flag" == i["label"].lower(): continue
        points = i["points"]
        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        area_list.append(tmp_mask.sum())
        valid_poly_list.append(i)

    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    for s_idx in list(sort_index):
        i = valid_poly_list[s_idx]
        label_value = 255 if "ignore" in i["label"].lower() else 1
        cv2.polylines(mask, np.array([i["points"]], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([i["points"]], dtype=np.int32), label_value)
    return mask, comments, is_sentence

def intersectionAndUnion(output, target, K, ignore_index=255):
    output = output.reshape(-1); target = target.reshape(-1)
    mask = target != ignore_index
    output = output[mask]; target = target[mask]
    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

class AverageMeter(object):
    def __init__(self, name): 
        self.name = name
        self.reset()
    def reset(self): 
        self.sum = 0
        self.count = 0
    def update(self, val, n=1): 
        self.sum += val * n
        self.count += n
    @property
    def avg(self): 
        return self.sum / (self.count + 1e-10)
    def state_dict(self): return {'sum': self.sum, 'count': self.count}
    def load_state_dict(self, state): self.sum = state['sum']; self.count = state['count']

class ExperimentLogger:
    def __init__(self, log_dir, fname, resume=False):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, f"{fname}.log")
        mode = 'a' if resume and os.path.exists(self.filepath) else 'w'
        self.start_time = datetime.datetime.now()
        with open(self.filepath, mode, encoding='utf-8') as f:
            if not resume:
                f.write(f"=== Log for {fname} ===\n")
                f.write(f"Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 50 + "\n")
            else:
                f.write(f"\n\n>>> Resumed at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} <<<\n")
                f.write("-" * 50 + "\n")

    def log(self, phase, content):
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        entry = f"[{timestamp}] [{phase}] {content}"
        # print(entry) # Removed stdout printing as per user request
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(entry + "\n")

    def log_raw(self, content):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(content + "\n")

class MaskRenderer:
    def __init__(self): self.default_color = (0, 255, 0); self.alpha = 0.5
    
    def overlay(self, img_bytes, mask, color=None, box=None):
        target_color = color if color else self.default_color
        try:
            while mask.ndim > 2: mask = mask.squeeze(0)
            orig = Image.open(BytesIO(img_bytes)).convert("RGB")
            m_pil = Image.fromarray(mask.astype(np.uint8)*255).resize(orig.size, Image.NEAREST)
            layer = Image.new('RGBA', orig.size, (*target_color, int(255*self.alpha)))
            final_m = Image.fromarray((np.array(m_pil)>0).astype(np.uint8)*255)
            applied = Image.new('RGBA', orig.size, (0,0,0,0)); applied.paste(layer, (0,0), final_m)
            comp = Image.alpha_composite(orig.convert('RGBA'), applied)

            if box:
                draw = ImageDraw.Draw(comp)
                draw.rectangle(box, outline=target_color, width=4)
                
            return comp
        except: return Image.open(BytesIO(img_bytes)).convert("RGB")

    def to_bytes(self, pil_img): 
        b = BytesIO()
        pil_img.convert("RGB").save(b, format='JPEG') 
        return b.getvalue()

def save_checkpoint(path, index, meters_dict, split):
    state = {
        "index": index,
        "split": split,
        "meters_state": {}
    }
    for group_key, group_meters in meters_dict.items():
        state["meters_state"][group_key] = {}
        for m_key, meter in group_meters.items():
            state["meters_state"][group_key][m_key] = meter.state_dict()
            
    with open(path, "w") as f:
        json.dump(state, f, indent=4)

def load_checkpoint(path, meters_dict):
    if not os.path.exists(path): return 0
    try:
        with open(path, "r") as f: state = json.load(f)
        start_index = state.get("index", 0) + 1
        saved_meters = state.get("meters_state", {})
        for group_key, group_saved in saved_meters.items():
            if group_key in meters_dict:
                for m_key, m_state in group_saved.items():
                    if m_key in meters_dict[group_key]:
                        meters_dict[group_key][m_key].load_state_dict(m_state)
        print(f"[Checkpoint] Resuming from index {start_index}.")
        return start_index
    except: return 0


def get_high_contrast_color(image_np: np.ndarray, mask: np.ndarray = None) -> tuple[tuple[int, int, int], str]:

    candidates = [
        ((255, 0, 0), "red"),
        ((0, 255, 0), "green"),
        ((0, 0, 255), "blue"),
        ((255, 255, 0), "yellow"),
        ((0, 255, 255), "cyan"),
        ((255, 0, 255), "magenta")
    ]
    

    candidate_colors = np.array([c[0] for c in candidates], dtype=np.float32)

    if isinstance(image_np, Image.Image):
        image_np = np.array(image_np)

    if mask is not None and np.any(mask):
        while mask.ndim > 2:
            mask = mask.squeeze(0)
        try:
            pixels = image_np[mask > 0]
        except IndexError:
            mask = mask.squeeze()
            pixels = image_np[mask > 0]
    else:
        pixels = image_np.reshape(-1, 3)
    MAX_PIXELS = 5000
    if len(pixels) > MAX_PIXELS:
        step = len(pixels) // MAX_PIXELS
        pixels = pixels[::step]
    
    pixels = pixels.astype(np.float32)

    diff = pixels[:, np.newaxis, :] - candidate_colors[np.newaxis, :, :]

    dists = np.sqrt(np.sum(diff**2, axis=2))

    mean_dists = np.mean(dists, axis=0)

    best_idx = np.argmax(mean_dists)
    best_color = candidates[best_idx]

    return best_color
