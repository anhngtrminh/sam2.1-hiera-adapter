import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.ao.quantization as quant
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import psutil
import threading

# ------------------------------
# RAM monitoring utilities
# ------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
process = psutil.Process(os.getpid())

def get_ram_mb():
    return process.memory_info().rss / 1024 / 1024


class RAMMonitor:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.running = False
        self.peak = 0
        self.thread = None

    def _monitor(self):
        while self.running:
            ram = process.memory_info().rss
            if ram > self.peak:
                self.peak = ram
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.peak = process.memory_info().rss
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def peak_mb(self):
        return self.peak / 1024 / 1024

# ---- class definitions -------------------------------------------------------
# Class 11 = "prohibited_item" is reserved for future fine-tuning.
# The current model will not produce it, but the pipeline handles it correctly
# if it ever appears (e.g. after fine-tuning on a labelled prohibited-item set).
CLASS_NAMES = [
    "background",           # 0  - never drawn
    # "bin",                  # 1
    # "concrete_bricks_tiles",# 2
    # "soils",                # 3
    "green_waste_timber",   # 4
    # "plastic",              # 5
    # "plastic",              # 6
    # "metals_e_waste",       # 7
    # "non_recyclable_waste", # 8
    # "cardboard",            # 9
    # "plaster_board",        # 10
    # "paint_can",            # 11  <- fine-tune target
    # "needles_syringes",     # 12 <- fine-tune target
    # "dead_animal",          # 13 <- fine-tune target
]

# RGBA fill used for the semi-transparent mask overlay
CLASS_COLORS_RGBA = [
    (  0,   0,   0,   0),   #  0 background    - fully transparent
    # (255, 215,   0, 120),   #  1 bin
    # (220,  20,  60, 120),   #  2 concrete
    # ( 64,  64,  64, 120),   #  3 soils
    ( 34, 139,  34, 120),   #  4 timber
    # (255, 140,   0, 120),   #  5 plastic (hard)
    # (255, 140,   0, 120),   #  6 plastic (soft)
    # ( 70, 130, 180, 120),   #  7 metals/e-waste
    # (255,  20, 147, 120),   #  8 non-recyclable
    # (210, 180, 140, 120),   #  9 cardboard
    # (186,  85, 211, 120),   # 10 plaster board
    # (255,   0,   0, 180),   # 11 paint_can - vivid red, more opaque
    # (255,   0,   0, 180),   # 12 needles_syringes - vivid red, more opaque
    # (255,   0,   0, 180),   # 13 dead_animal - vivid red, more opaque
]

# Solid RGB for bbox borders and label chips
CLASS_COLORS_RGB = [(r, g, b) for r, g, b, _ in CLASS_COLORS_RGBA]

# Any class whose name appears here is flagged as prohibited in the output JSON.
# Extend this set as the label space grows.
PROHIBITED_NAMES = {"paint_can", "needles_syringes", "dead_animal"}

# Classes that are never drawn, annotated, or exported.
# 0 = background; also covers the binary-mask "foreground" index when the model
# outputs a single-channel mask and pred_to_mask returns 0/1 with 0=background.
IGNORED_CLASSES = {0}

# ---- device ------------------------------------------------------------------
# DEVICE = torch.device("cpu")
print(f"[demo_cpu] Running on device: {device}")

# ---- config / model helpers --------------------------------------------------

def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg, ckpt_path=None):
    """Build the SAM wrapper exactly as train.py / test.py do.

    num_classes is injected in two ways (first wins):
      1. From the checkpoint weights — reads cls_upscaling.3.weight shape so the
         model architecture always matches the saved file, regardless of config.
      2. From cfg["classes"] list (config YAML) — len(classes) - 1 foreground + bg.
      3. Fallback to cfg["model"]["args"]["num_classes"] if explicitly set.
    """
    try:
        from models import make as make_model
    except ImportError:
        raise ImportError(
            "Cannot import 'models'. "
            "Run this script from inside the SAM2-Adapter-CDW repo root."
        )

    # ── Infer num_classes ────────────────────────────────────────────────────
    num_classes = None

    # Priority 1: read directly from checkpoint weight shape
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            import torch as _torch
            _sd  = _torch.load(ckpt_path, map_location="cpu")
            if isinstance(_sd, dict):
                for _k in ("model", "state_dict", "net", "model_state_dict"):
                    if _k in _sd:
                        _sd = _sd[_k]
                        break
            _key = "mask_decoder.cls_upscaling.3.weight"
            if _key in _sd:
                # output channels = 32 * num_classes  (transformer_dim*num_classes//8, 64//8=32*nc)
                num_classes = _sd[_key].shape[0] // 32
                print(f"[demo] num_classes={num_classes} detected from checkpoint.")
        except Exception as _e:
            print(f"[demo] Could not read num_classes from checkpoint: {_e}")

    # Priority 2: count from config classes list
    if num_classes is None and cfg.get("classes"):
        num_classes = len(cfg["classes"])   # includes background at index 0
        print(f"[demo] num_classes={num_classes} inferred from config classes list.")

    # Priority 3: already set in model args
    if num_classes is None:
        num_classes = cfg.get("model", {}).get("args", {}).get("num_classes")

    if num_classes is None:
        raise ValueError(
            "Cannot determine num_classes. Pass --model <ckpt> so it can be "
            "read from the checkpoint, or set 'classes' in your config YAML."
        )

    cfg.setdefault("model", {}).setdefault("args", {})["num_classes"] = num_classes
    return make_model(cfg["model"])


def load_weights(model, ckpt_path):
    print(f"[demo_cpu] Loading weights from: {ckpt_path}")
    # if "int8" in ckpt_path.lower():
    #     for name, module in model.named_modules():
    #         if "image_encoder" in name:
    #             module.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
    #         else:
    #             module.qconfig = None

    #     quant.prepare(model, inplace=True)
    #     quant.convert(model, inplace=True)

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict):
        for key in ("model", "state_dict", "net", "model_state_dict"):
            if key in state:
                state = state[key]
                break

    # cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    cleaned = {}

    for k, v in state.items():
        k = k.replace("module.", "")

        # FIX: convert quantized tensor → float tensor
        if isinstance(v, torch.Tensor) and v.is_quantized:
            v = v.dequantize()

        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    if missing:
        print(f"[demo_cpu] WARNING - missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[demo_cpu] WARNING - unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    print("[demo_cpu] Weights loaded OK")
    return model

# ---- image pre-processing ----------------------------------------------------

IMG_SIZE = 1024   # SAM2 native; use --size 512 for faster CPU runs

def get_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def preprocess(img_path, size, prescale):
    from PIL import ImageOps
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    print(f"[demo_cpu] Original size : W={img.width}  H={img.height}")
    if prescale > 0 and max(img.width, img.height) > prescale:
        img.thumbnail((prescale, prescale), Image.LANCZOS)
        print(f"[demo_cpu] Prescaled to  : W={img.width}  H={img.height}")
    tensor = get_transform(size)(img).unsqueeze(0).to(device)
    return tensor, img.copy()

# ---- model dispatch ----------------------------------------------------------

def run_model(model, tensor):
    dummy_gt = torch.zeros(
        tensor.shape[0], 1, tensor.shape[2], tensor.shape[3], device=device
    )
    if hasattr(model, "set_input") and hasattr(model, "infer"):
        model.set_input(tensor, dummy_gt)
        return model.infer(tensor)
    if hasattr(model, "set_input"):
        model.set_input(tensor, dummy_gt)
        return model.forward()
    if hasattr(model, "forward_pass"):
        return model.forward_pass(tensor)
    import inspect
    sig    = inspect.signature(model.forward)
    params = [p for p in sig.parameters if p != "self"]
    if len(params) == 0:
        model.input = tensor
        return model.forward()
    raise RuntimeError(
        f"Cannot determine how to call {type(model).__name__}.forward().\n"
        f"Signature: {sig}\n"
        "Please check models/sam.py and update run_model() in demo_cpu.py."
    )

# ---- mask decoding -----------------------------------------------------------

def pred_to_mask(output):
    """Raw model output -> integer H x W class-index array (uint8)."""
    if isinstance(output, (list, tuple)):
        output = output[0]
    if output.dim() == 4:
        output = output.squeeze(0)
    if output.shape[0] > 1:
        return output.argmax(0).detach().cpu().numpy().astype(np.uint8)
    return (torch.sigmoid(output.squeeze(0)) > 0.5).detach().cpu().numpy().astype(np.uint8)

# ---- per-class region helpers ------------------------------------------------

def get_bboxes(mask, orig_size, min_area_ratio=0.0001):
    """
    Compute a tight bounding box for every individual connected-component
    instance of each non-background class, scaled to original image dimensions.

    Uses scipy.ndimage.label for connected-component analysis so that two
    separate blobs of the same class each get their own box.

    Returns:
        list of dicts, one per instance:
        [
            {"cls_id": int, "x1": int, "y1": int, "x2": int, "y2": int,
             "pixel_count": int},
            ...
        ]
    """
    from scipy.ndimage import label as cc_label

    mask_h, mask_w = mask.shape
    orig_w, orig_h = orig_size
    sx = orig_w / mask_w
    sy = orig_h / mask_h
    min_pixels = mask_h * mask_w * min_area_ratio

    instances = []
    for cls_id in np.unique(mask):
        if cls_id in IGNORED_CLASSES:
            continue

        binary        = (mask == cls_id).astype(np.uint8)
        labeled, n_cc = cc_label(binary)           # label each blob independently

        for cc_idx in range(1, n_cc + 1):
            ys, xs = np.where(labeled == cc_idx)
            if len(xs) < min_pixels:               # skip tiny noise blobs
                continue
            instances.append({
                "cls_id":      int(cls_id),
                "x1":          int(xs.min() * sx),
                "y1":          int(ys.min() * sy),
                "x2":          int(xs.max() * sx),
                "y2":          int(ys.max() * sy),
                "pixel_count": int(len(xs)),
            })

    return instances


def mask_to_polygons(mask, orig_size, epsilon_factor=0.002, min_area_ratio=0.0001):
    """
    Extract simplified contour polygons per class using Douglas-Peucker.

    Returns: { cls_id: [ [[x,y], ...], ... ] }
    """
    try:
        import cv2
        _cv2 = True
    except ImportError:
        _cv2 = False

    orig_w, orig_h = orig_size
    mask_h, mask_w = mask.shape
    sx = orig_w / mask_w
    sy = orig_h / mask_h
    min_area = mask_w * mask_h * min_area_ratio

    result = {}

    for cls_id in np.unique(mask):
        if cls_id in IGNORED_CLASSES:
            continue
        binary = (mask == cls_id).astype(np.uint8) * 255
        polys  = []

        if _cv2:
            import cv2
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
            )
            for cnt in contours:
                if cv2.contourArea(cnt) < min_area:
                    continue
                peri   = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
                if len(approx) >= 3:
                    polys.append([
                        [round(float(p[0][0]) * sx, 2), round(float(p[0][1]) * sy, 2)]
                        for p in approx
                    ])
        else:
            try:
                from skimage import measure
                from skimage.measure import approximate_polygon
            except ImportError:
                print("[demo_cpu] WARNING: Neither cv2 nor skimage available; "
                      "polygon export skipped.")
                break
            padded   = np.pad(binary, 1)
            contours = measure.find_contours(padded, level=127.5)
            for cnt in contours:
                area = 0.5 * abs(
                    np.dot(cnt[:, 1], np.roll(cnt[:, 0], 1))
                    - np.dot(cnt[:, 0], np.roll(cnt[:, 1], 1))
                )
                if area < min_area:
                    continue
                peri       = np.sum(np.linalg.norm(np.diff(cnt, axis=0), axis=1))
                simplified = approximate_polygon(cnt, tolerance=max(epsilon_factor * peri, 1.0))
                if len(simplified) >= 3:
                    polys.append([
                        [round((float(p[1]) - 1) * sx, 2), round((float(p[0]) - 1) * sy, 2)]
                        for p in simplified
                    ])

        if polys:
            result[int(cls_id)] = polys

    return result

# ---- visualisation -----------------------------------------------------------

def render_visualization(orig, mask, bboxes):
    """
    Produce the output visualisation image:
      - Background (class 0) kept exactly as the original photo — untouched
      - Foreground class pixels get a semi-transparent colour fill composited
        on top of the original image
      - Tight bounding box per *instance* (one box per connected component,
        not one box per class)
      - Inline label chip above each bbox: class name + coverage %
        (prohibited_item chips show a [!] warning prefix)

    bboxes: list of instance dicts returned by get_bboxes()
            [{"cls_id", "x1","y1","x2","y2", "pixel_count"}, ...]
    """
    orig_w, orig_h = orig.size

    # 1. Scale mask to original resolution
    mask_pil = Image.fromarray(mask).resize((orig_w, orig_h), Image.NEAREST)
    mask_np  = np.array(mask_pil)

    # 2. Start from the unmodified original image
    base = orig.convert("RGB")

    # 3. Build a per-class colour overlay (only foreground classes; class 0 transparent)
    overlay = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))
    for cls_id in np.unique(mask_np):
        if cls_id == 0:
            continue
        r, g, b, a = CLASS_COLORS_RGBA[cls_id % len(CLASS_COLORS_RGBA)]
        region = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        region[mask_np == cls_id] = [r, g, b, a]
        overlay = Image.alpha_composite(overlay, Image.fromarray(region, mode="RGBA"))

    # Composite overlay onto original; background pixels are unaffected (alpha=0)
    canvas = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

    # 4. Bounding boxes + inline label chips — one per instance
    draw = ImageDraw.Draw(canvas)
    try:
        font_label = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font_label = ImageFont.load_default()

    total_pixels = mask_np.size

    for inst in bboxes:
        cls_id = inst["cls_id"]
        x1, y1, x2, y2 = inst["x1"], inst["y1"], inst["x2"], inst["y2"]
        color = CLASS_COLORS_RGB[cls_id % len(CLASS_COLORS_RGB)]
        name  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        pct   = inst["pixel_count"] / total_pixels * 100
        is_prohibited = name in PROHIBITED_NAMES

        # Bounding box border (thicker red outline for prohibited items)
        border_color = (220, 20, 20) if is_prohibited else color
        border_width = 3 if is_prohibited else 2
        draw.rectangle([x1, y1, x2, y2], outline=border_color, width=border_width)

        # Label chip
        prefix     = "[!] " if is_prohibited else ""
        label_text = f"{prefix}{name}  {pct:.1f}%"
        try:
            tb = draw.textbbox((0, 0), label_text, font=font_label)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except AttributeError:
            tw, th = draw.textsize(label_text, font=font_label)

        pad    = 4
        chip_x = x1
        chip_y = max(y1 - th - pad * 2 - 2, 0)

        chip_bg = Image.new("RGBA", (tw + pad * 2, th + pad * 2), color + (210,))
        canvas.paste(chip_bg.convert("RGB"), (chip_x, chip_y), mask=chip_bg.split()[3])
        draw.text(
            (chip_x + pad, chip_y + pad),
            label_text,
            fill=(255, 255, 255),
            font=font_label,
        )

    return canvas

# ---- JSON result builder -----------------------------------------------------

def build_result_json(img_path, orig_size, mask, bboxes, polygons, inference_time_s):
    """
    Build the structured result dictionary for one image.

    bboxes: list of instance dicts from get_bboxes()
            [{"cls_id", "x1","y1","x2","y2","pixel_count"}, ...]
    """
    orig_w, orig_h = orig_size
    mask_np    = np.array(Image.fromarray(mask).resize((orig_w, orig_h), Image.NEAREST))
    total_px   = mask_np.size

    present_ids    = sorted(int(c) for c in np.unique(mask_np) if c not in IGNORED_CLASSES)
    categories     = [
        CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        for c in present_ids
    ]

    # Build one detection entry per instance (not per class)
    detections = []
    for inst in bboxes:
        cls_id   = inst["cls_id"]
        label    = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        coverage = round(inst["pixel_count"] / total_px * 100, 2)
        detections.append({
            "cls_id":       cls_id,
            "label":        label,
            "prohibited":   label in PROHIBITED_NAMES,
            "coverage_pct": coverage,
            "bbox":         {k: inst[k] for k in ("x1", "y1", "x2", "y2")},
            "polygons":     polygons.get(cls_id, []),
        })

    prohibited_items = list({d["label"] for d in detections if d["prohibited"]})

    if prohibited_items:
        reason = "Prohibited item(s) detected."
    elif present_ids:
        reason = "No prohibited items detected."
    else:
        reason = "No foreground objects detected."

    return {
        "success":          len(prohibited_items) == 0,
        "reason":           reason,
        "imagePath":        Path(img_path).name,
        "imageWidth":       orig_w,
        "imageHeight":      orig_h,
        "inferenceTime_s":  round(inference_time_s, 3),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "categoriesList":   categories,
        "prohibitedItems":  prohibited_items,
        # "detections":       detections,
    }


def export_json(result, img_path, out_dir):
    """Write result dict to <stem>.json in out_dir."""
    stem     = Path(img_path).stem
    out_path = os.path.join(out_dir, f"{stem}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, separators=(",", ": "))
    n    = len(result.get("detections", []))
    flag = "  *** PROHIBITED ***" if result.get("prohibitedItem") else ""
    print(f"[demo_cpu] Saved JSON    -> {out_path}  ({n} detection(s)){flag}")
    return out_path

# ---- single-image inference --------------------------------------------------

@torch.no_grad()
def infer_single(model, img_path, out_dir, size, prescale,
                 epsilon_factor=0.002, min_area_ratio=0.0001):
    # measure memory + time
    ram_before = get_ram_mb()
    monitor = RAMMonitor()

    t0 = time.time()

    monitor.start()
                
    model.eval()
    model.to(device)

    tensor, orig = preprocess(img_path, size, prescale)

    print(f"[demo_cpu] Running inference on {img_path} ...")
    t0 = time.time()
    output = run_model(model, tensor)
    dt = time.time() - t0
    print(f"[demo_cpu] Done in {dt:.1f}s")

    mask = pred_to_mask(output)
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(img_path).stem

    # Bounding boxes + polygons (mask space -> original image space)
    bboxes   = get_bboxes(mask, orig.size)
    polygons = mask_to_polygons(mask, orig.size, epsilon_factor, min_area_ratio)

    monitor.stop()

    t1 = time.time()
    ram_after = get_ram_mb()

    print("----- Inference Stats -----")
    print(f"Inference time: {t1 - t0:.3f} sec")
    print(f"RAM before: {ram_before:.2f} MB")
    print(f"RAM after: {ram_after:.2f} MB")
    print(f"RAM used: {ram_after - ram_before:.2f} MB")
    print(f"Peak RAM during inference: {monitor.peak_mb():.2f} MB")
    print("---------------------------")

    # Visualisation: white bg + overlay + bbox + inline labels (no mask PNG, no legend)
    vis      = render_visualization(orig, mask, bboxes)
    vis_path = os.path.join(out_dir, f"{stem}_vis.png")
    vis.save(vis_path)
    print(f"[demo_cpu] Saved vis     -> {vis_path}")

    # Structured JSON result
    result = build_result_json(img_path, orig.size, mask, bboxes, polygons, dt)
    export_json(result, img_path, out_dir)

    # Console summary
    # print("[demo_cpu] Detected classes:")
    # for d in result["detections"]:
    #     flag = "  [!] PROHIBITED" if d["prohibited"] else ""
    #     print(f"           [{d['cls_id']:2d}] {d['label']:<22s}  {d['coverage_pct']:.1f}%{flag}")
    # if result["prohibitedItems"]:
    #     print("[demo_cpu] *** PROHIBITED ITEMS: " + ", ".join(result["prohibitedItems"]) + " ***")

# ---- dataset-level evaluation ------------------------------------------------

@torch.no_grad()
def infer_dataset(model, cfg, out_dir, size,
                  epsilon_factor=0.002, min_area_ratio=0.0001):
    model.eval()
    model.to(device)

    try:
        from datasets import make as make_dataset
    except ImportError:
        print("[demo_cpu] ERROR: Cannot import 'datasets'. Run from repo root.")
        sys.exit(1)

    ds_cfg  = cfg.get("val_dataset", cfg.get("test_dataset", {}))
    dataset = make_dataset(ds_cfg)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )

    os.makedirs(out_dir, exist_ok=True)
    total_iou, n = 0.0, 0

    for i, batch in enumerate(loader):
        inp = batch["inp"].to(device)
        gt  = batch.get("gt", None)

        t0  = time.time()
        out = run_model(model, inp)
        dt  = time.time() - t0

        mask = pred_to_mask(out)

        # Binary IoU
        iou = None
        if gt is not None:
            gt_b  = (gt.squeeze().cpu().numpy() > 0).astype(np.uint8)
            pr_b  = (mask > 0).astype(np.uint8)
            inter = (pr_b & gt_b).sum()
            union = (pr_b | gt_b).sum()
            iou   = inter / (union + 1e-6)
            total_iou += iou
            n += 1

        name      = batch.get("name", [f"sample_{i:04d}"])[0]
        img_path  = batch.get("img_path", [f"{name}.jpg"])[0]
        orig_size = batch.get("orig_size", None)
        if orig_size is not None:
            orig_w, orig_h = int(orig_size[0]), int(orig_size[1])
        else:
            orig_w, orig_h = mask.shape[1], mask.shape[0]

        # Reconstruct PIL image for visualisation (white placeholder if unavailable)
        orig_img = batch.get("orig_img", None)
        if orig_img is not None:
            orig_pil = Image.fromarray(
                (orig_img.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            ).resize((orig_w, orig_h))
        else:
            orig_pil = Image.new("RGB", (orig_w, orig_h), (255, 255, 255))

        bboxes   = get_bboxes(mask, (orig_w, orig_h))
        polygons = mask_to_polygons(mask, (orig_w, orig_h), epsilon_factor, min_area_ratio)

        vis = render_visualization(orig_pil, mask, bboxes)
        vis.save(os.path.join(out_dir, f"{name}_vis.png"))

        result = build_result_json(img_path, (orig_w, orig_h), mask, bboxes, polygons, dt)
        export_json(result, img_path, out_dir)

        if (i + 1) % 10 == 0 or i == 0:
            iou_str = f"  IoU={iou:.4f}" if iou is not None else ""
            print(f"[demo_cpu] [{i+1}/{len(loader)}] {dt:.1f}s/img{iou_str}")

    if n > 0:
        print(f"\n[demo_cpu] Mean IoU over {n} samples: {total_iou/n:.4f}")

# ---- CLI ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CPU demo for SAM2-Adapter-CDW")
    p.add_argument("--config",  default="configs/train.yaml",
                   help="Path to YAML config file")
    p.add_argument("--model",   default="C:/Users/ntmanh1/Documents/project/handel group/segmentation/28573229/SAM2-Adapter-CDW/streamlit/checkpoints/model_epoch_best.pth",
                   help="Path to trained .pth checkpoint")

    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--img",     default="C:/Users/ntmanh1/Downloads/image (1).png",
                     help="Single image path")
    grp.add_argument("--folder",  default=None,
                     help="Folder of images (jpg/png/jpeg); runs all inside it")
    grp.add_argument("--dataset", action="store_true",
                     help="Run on the full test split from the config")

    p.add_argument("--prescale", type=int,   default=1024,
                   help="Downscale longer edge to N px before inference (0=off)")
    p.add_argument("--out",      default="output_demo",
                   help="Output directory for _vis.png files + JSON")
    p.add_argument("--size",     type=int,   default=IMG_SIZE,
                   help=f"Model input resolution (default: {IMG_SIZE})")
    p.add_argument("--epsilon",  type=float, default=0.002,
                   help="Douglas-Peucker simplification factor (higher = fewer polygon points)")
    p.add_argument("--min-area", type=float, default=0.0001,
                   help="Min contour area as fraction of image area (filters noise blobs)")
    return p.parse_args()


def collect_images(folder):
    """Return sorted list of image paths inside folder (non-recursive)."""
    exts  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    paths = sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if not paths:
        print(f"[demo_cpu] WARNING: no images found in {folder}")
    return [str(p) for p in paths]


def main():
    args  = parse_args()
    cfg   = load_config(args.config)
    model = build_model(cfg, ckpt_path=args.model)
    model = load_weights(model, args.model)

    kw = dict(epsilon_factor=args.epsilon, min_area_ratio=args.min_area)

    if args.folder:
        img_paths = collect_images(args.folder)
        total     = len(img_paths)
        print(f"[demo_cpu] Folder mode: {total} image(s) in {args.folder}")
        for i, img_path in enumerate(img_paths, 1):
            print(f"\n[demo_cpu] [{i}/{total}] {img_path}")
            try:
                infer_single(model, img_path, args.out, args.size, args.prescale, **kw)

            except Exception as e:
                print(f"[demo_cpu] ERROR on {img_path}: {e}")
        print(f"[demo_cpu] Folder done - {total} image(s) processed -> {args.out}/")

    elif args.dataset:
        print("[demo_cpu] Dataset evaluation ...")
        infer_dataset(model, cfg, args.out, args.size, **kw)

    else:
        print(f"[demo_cpu] Single-image demo -> {args.img}")
        infer_single(model, args.img, args.out, args.size, args.prescale, **kw)

    print("[demo_cpu] Done")


if __name__ == "__main__":
    main()