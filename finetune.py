"""
train.py  —  CDW-Seg  |  COCO-format training script
=====================================================
Workflow
--------
1.  prepare_coco_splits()  — run once to restructure your flat dataset folder
    into the standard COCO layout and generate split annotation JSON files:

        <coco_root>/
            images/
                train/
                val/
                test/
            annotations/
                instances_train.json
                instances_val.json
                instances_test.json

    Source folder must contain:
        <source_dir>/
            <source_images_subdir>/   ← image files  (.jpg / .png / ...)
            instances_all.json        ← single COCO JSON covering all images
         OR
            <source_annotations_subdir>/  ← one JSON per image

    Run standalone (once before training):
        python train.py --config configs/train.yaml --prepare-only

    Re-run if source data changes:
        python train.py --config configs/train.yaml --prepare-only --force-prepare

2.  Training reads the restructured layout via COCOSegDataset:
        python train.py --config configs/train.yaml --name my_run

    Multi-GPU DDP:
        torchrun --nproc_per_node=2 train.py --config configs/train.yaml --name my_run
"""

import argparse
import json
import math
import os
import random
import shutil
from statistics import mean
from xml.parsers.expat import model

import numpy as np
import torch
import torch.distributed as dist
import yaml
from PIL import Image
from prettytable import PrettyTable
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T
from tqdm import tqdm

import models
import utils
from eval_iou import SegmentationMetric

# ── Globals (set in main) ──────────────────────────────────────────────────────
config     = None
log        = None
writer     = None
local_rank = 0
is_main    = True


# =============================================================================
# Step 1 — Flat folder -> COCO split layout
# =============================================================================

def prepare_coco_splits(cfg: dict, force: bool = False) -> str:
    """
    Restructure a flat dataset folder into train / val / test COCO splits.

    Source layout (any of the following are accepted):

      A) Single combined JSON
            source_dir/
                images/            ← image files
                instances_all.json ← COCO JSON for all images  (or set source_ann_file)

      B) Per-image JSON files
            source_dir/
                images/
                annotations/       ← one .json per image, named <stem>.json

      C) Images only (no annotations)
            source_dir/
                images/
            -> produces empty annotation files; add annotations later.

    Output:
        coco_root/
            images/train/   val/   test/
            annotations/instances_train.json  instances_val.json  instances_test.json

    Returns the coco_root path.
    """
    sp     = cfg['split_preparation']
    src    = sp['source_dir']
    root   = cfg['coco']['coco_root']
    splits = ('train', 'val', 'test')

    ann_dir   = os.path.join(root, 'annotations')
    img_dirs  = {s: os.path.join(root, 'images', s) for s in splits}
    ann_paths = {s: os.path.join(ann_dir, f'instances_{s}.json') for s in splits}

    if not force and all(os.path.isfile(ann_paths[s]) for s in splits):
        print(f'[prepare] Splits already exist at {root}  (use --force-prepare to redo).')
        return root

    print(f'[prepare] Source : {src}')
    print(f'[prepare] Output : {root}')

    for d in list(img_dirs.values()) + [ann_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Find source images ────────────────────────────────────────────────────
    src_img_dir = os.path.join(src, sp.get('source_images_subdir', 'images'))
    if not os.path.isdir(src_img_dir):
        src_img_dir = src          # images live directly in source_dir

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    all_imgs = sorted(
        f for f in os.listdir(src_img_dir)
        if os.path.splitext(f)[1].lower() in img_exts
    )
    if not all_imgs:
        raise RuntimeError(f'No images found in {src_img_dir}')
    print(f'[prepare] Found {len(all_imgs)} images.')

    # ── Load or build a combined COCO annotation dict ─────────────────────────
    combined = _load_source_annotations(src, sp, src_img_dir, all_imgs, cfg)

    # ── Shuffle + split by ratio ──────────────────────────────────────────────
    rng = random.Random(sp.get('split_seed', 42))
    idx = list(range(len(combined['images'])))
    rng.shuffle(idx)

    n        = len(idx)
    n_train  = math.floor(n * sp.get('train_ratio', 0.80))
    n_val    = math.floor(n * sp.get('val_ratio',   0.10))
    split_idx = {
        'train': idx[:n_train],
        'val':   idx[n_train: n_train + n_val],
        'test':  idx[n_train + n_val:],
    }

    # ── Write per-split JSON and copy images ──────────────────────────────────
    for split, idxs in split_idx.items():
        imgs       = [combined['images'][i] for i in idxs]
        id_set     = {im['id'] for im in imgs}
        anns       = [a for a in combined['annotations'] if a['image_id'] in id_set]

        coco_split = {
            'info':        combined.get('info', {}),
            'licenses':    combined.get('licenses', []),
            'categories':  combined['categories'],
            'images':      imgs,
            'annotations': anns,
        }
        with open(ann_paths[split], 'w') as f:
            json.dump(coco_split, f)
        print(f'[prepare]  {split:5s}: {len(imgs):4d} images, '
              f'{len(anns):5d} annotations -> {ann_paths[split]}')

        for im in tqdm(imgs, desc=f'copying {split}', leave=False):
            dst = os.path.join(img_dirs[split], im['file_name'])
            if not os.path.isfile(dst):
                shutil.copy2(os.path.join(src_img_dir, im['file_name']), dst)

    print(f'[prepare] Done. COCO dataset ready at: {root}')
    return root


def _load_source_annotations(src: str, sp: dict, src_img_dir: str,
                              all_imgs: list, cfg: dict) -> dict:
    """Return a single combined COCO dict (all images, all annotations).

    Detection order:
      1. Explicit source_ann_file in config
      2. <src>/instances_all.json
      3. <src>/annotations.json          (standard single-file COCO export)
      4. Any single .json in <src>/annotations/ subfolder
      5. Per-image JSONs in annotations/ subfolder (one file per image)
      6. Images-only skeleton (no annotations found)
    """

    # ── Option A: explicit or well-known combined JSON ────────────────────────
    # Check in priority order: config override, instances_all.json, annotations.json
    ann_subdir      = os.path.join(src, sp.get('source_annotations_subdir', 'annotations'))
    candidates      = [
        sp.get('source_ann_file'),                        # explicit override
        os.path.join(src, 'instances_all.json'),          # common export name
        os.path.join(src, 'annotations.json'),            # standard COCO name
        os.path.join(ann_subdir, 'annotations.json'),     # inside annotations/
        os.path.join(ann_subdir, 'instances_all.json'),   # inside annotations/
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            print(f'[prepare] Loading combined annotation: {candidate}')
            with open(candidate, encoding='utf-8') as f:
                data = json.load(f)
            # Only keep images whose file is actually present on disk
            present = set(all_imgs)
            data['images'] = [im for im in data['images'] if im['file_name'] in present]
            id_set = {im['id'] for im in data['images']}
            data['annotations'] = [a for a in data['annotations'] if a['image_id'] in id_set]
            print(f'[prepare]  Found {len(data["images"])} images, '
                  f'{len(data["annotations"])} annotations in combined JSON.')
            return data

    # ── Option B: single JSON in annotations/ that is not per-image ──────────
    # If the subfolder exists and contains exactly one JSON, treat it as combined.
    if os.path.isdir(ann_subdir):
        jsons = [f for f in os.listdir(ann_subdir) if f.endswith('.json')]
        if len(jsons) == 1:
            candidate = os.path.join(ann_subdir, jsons[0])
            print(f'[prepare] Loading single JSON from annotations/: {candidate}')
            with open(candidate, encoding='utf-8') as f:
                data = json.load(f)
            # Validate it looks like a COCO file (has images + annotations keys)
            if 'images' in data and 'annotations' in data:
                present = set(all_imgs)
                data['images'] = [im for im in data['images'] if im['file_name'] in present]
                id_set = {im['id'] for im in data['images']}
                data['annotations'] = [a for a in data['annotations'] if a['image_id'] in id_set]
                print(f'[prepare]  Found {len(data["images"])} images, '
                      f'{len(data["annotations"])} annotations.')
                return data

    # ── Option C: per-image JSON files in annotations/ ────────────────────────
    if os.path.isdir(ann_subdir):
        json_stems = {os.path.splitext(f)[0] for f in os.listdir(ann_subdir)
                      if f.endswith('.json')}
        img_stems  = {os.path.splitext(f)[0] for f in all_imgs}
        if json_stems & img_stems:   # at least some overlap → per-image format
            print(f'[prepare] Merging per-image JSONs from: {ann_subdir}')
            return _merge_per_image_jsons(ann_subdir, src_img_dir, all_imgs, cfg)

    # ── Option D: images only — empty annotations ─────────────────────────────
    print('[prepare] WARNING: no annotations found — building image-only skeleton.')
    images = []
    for i, fname in enumerate(all_imgs, start=1):
        pil = Image.open(os.path.join(src_img_dir, fname))
        w, h = pil.size
        images.append({'id': i, 'file_name': fname, 'width': w, 'height': h})
    return {
        'info': {}, 'licenses': [],
        'categories': _categories_from_cfg(cfg),
        'images': images, 'annotations': [],
    }


def _merge_per_image_jsons(ann_dir: str, img_dir: str,
                            all_imgs: list, cfg: dict) -> dict:
    """Merge one-JSON-per-image files into a single combined COCO dict."""
    categories = None
    images, anns = [], []
    img_id = ann_id = 1
    stems  = {os.path.splitext(f)[0]: f for f in all_imgs}

    for stem, fname in sorted(stems.items()):
        jpath = os.path.join(ann_dir, stem + '.json')
        if not os.path.isfile(jpath):
            continue
        with open(jpath) as f:
            data = json.load(f)

        if categories is None:
            categories = data.get('categories')

        im_info  = data['images'][0] if data.get('images') else {}
        orig_id  = im_info.get('id', 1)
        w        = im_info.get('width') or Image.open(os.path.join(img_dir, fname)).size[0]
        h        = im_info.get('height') or Image.open(os.path.join(img_dir, fname)).size[1]

        images.append({'id': img_id, 'file_name': fname, 'width': w, 'height': h})
        for ann in data.get('annotations', []):
            if ann.get('image_id') == orig_id:
                new_ann = {**ann, 'id': ann_id, 'image_id': img_id}
                anns.append(new_ann)
                ann_id += 1
        img_id += 1

    return {
        'info': {}, 'licenses': [],
        'categories': categories or _categories_from_cfg(cfg),
        'images': images, 'annotations': anns,
    }


def _categories_from_cfg(cfg: dict) -> list:
    """Build COCO categories from the classes list (skip index-0 background)."""
    return [
        {'id': i, 'name': name, 'supercategory': 'waste'}
        for i, name in enumerate(cfg['classes'])
        if i > 0
    ]


# =============================================================================
# Step 2 — COCO Semantic Segmentation Dataset
# =============================================================================

class COCOSegDataset(Dataset):
    """
    Reads a COCO-layout split produced by prepare_coco_splits() and returns:
        inp : float32  (3, H, W)  — normalised image tensor
        gt  : int64    (H, W)     — per-pixel class index (0 = background)

    Class index mapping is derived directly from the JSON annotation file:
        index 0          -> background  (pixels covered by no annotation)
        index 1 .. N     -> categories sorted by their id in the JSON

    This means num_classes = number of JSON categories + 1, and the YAML
    classes list is used only for logging / metric labels — not for ID mapping.

    Instance masks are merged into one semantic mask; smaller instances are
    drawn last so they appear on top of larger ones.
    """

    def __init__(
        self,
        coco_root: str,
        split:     str,
        ann_file:  str  = None,
        inp_size:  int  = 1024,
        augment:   bool = False,
        ignore_bg: bool = False,
        **_,                      # absorbs unused kwargs (classes, etc.)
    ):
        try:
            from pycocotools.coco import COCO
            from pycocotools import mask as coco_mask
        except ImportError:
            raise ImportError('pip install pycocotools')

        self.inp_size  = inp_size
        self.augment   = augment
        self.ignore_bg = ignore_bg
        self.coco_mask = coco_mask

        if ann_file is None:
            ann_file = os.path.join('annotations', f'instances_{split}.json')
        ann_path = os.path.join(coco_root, ann_file)
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f'Annotation not found: {ann_path}')

        self.coco    = COCO(ann_path)
        self.img_dir = os.path.join(coco_root, 'images', split)

        # ── Build category_id -> contiguous class index from the JSON itself ──
        # JSON categories sorted by their own id field (ascending).
        # Index 0 is always background; JSON categories start at index 1.
        #
        # Example from instances_train.json:
        #   {"id": 1, "name": "bin"}         -> class index 1
        #   {"id": 2, "name": "concrete_bricks_tiles"} -> class index 2
        #   ...
        #   {"id": 9, "name": "plaster_board"} -> class index 9
        #
        # num_classes = 10  (0=background + 9 JSON categories)
        sorted_cats     = sorted(self.coco.loadCats(self.coco.getCatIds()),
                                 key=lambda c: c['id'])
        self.cat_to_cls = {cat['id']: idx + 1
                           for idx, cat in enumerate(sorted_cats)}
        self.num_classes = len(sorted_cats) + 1   # +1 for background
        self.class_names = ['background'] + [c['name'] for c in sorted_cats]

        if is_main:
            log(f'[{split}] {self.num_classes} classes (incl. background):')
            for cid, cidx in self.cat_to_cls.items():
                log(f'  JSON cat_id={cid} -> class_idx={cidx}'
                    f'  ({self.class_names[cidx]})')

        ann_img_ids   = {a['image_id'] for a in self.coco.dataset.get('annotations', [])}
        self.img_ids  = sorted(ann_img_ids & set(self.coco.getImgIds()))
        if not self.img_ids:
            raise RuntimeError(f'No annotated images in {ann_path}')

        self.img_tf = T.Compose([
            T.Resize((inp_size, inp_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        info = self.coco.loadImgs(self.img_ids[idx])[0]
        img  = Image.open(os.path.join(self.img_dir, info['file_name'])).convert('RGB')

        sem  = self._build_semantic_mask(self.img_ids[idx], info['height'], info['width'])
        mask = Image.fromarray(sem).resize((self.inp_size, self.inp_size), Image.NEAREST)

        if self.augment:
            img, mask = _augment(img, mask)

        inp  = self.img_tf(img)
        gt   = np.array(mask, dtype=np.int64)
        if self.ignore_bg:
            gt[gt == 0] = 255      # exclude background from loss

        return {'inp': inp, 'gt': torch.from_numpy(gt)}

    def _build_semantic_mask(self, img_id: int, h: int, w: int) -> np.ndarray:
        semantic = np.zeros((h, w), dtype=np.uint8)
        anns     = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        for ann in sorted(anns, key=lambda a: a.get('area', 0), reverse=True):
            cls = self.cat_to_cls.get(ann['category_id'], 0)
            if cls == 0:
                continue
            binary = _decode_coco_mask(ann, self.coco_mask, h, w)
            semantic[binary > 0] = cls
        return semantic


def _decode_coco_mask(ann: dict, coco_mask, h: int, w: int) -> np.ndarray:
    seg = ann.get('segmentation', {})
    if isinstance(seg, dict):
        binary = coco_mask.decode(coco_mask.frPyObjects(seg, h, w))
    elif isinstance(seg, list) and seg:
        binary = coco_mask.decode(coco_mask.merge(coco_mask.frPyObjects(seg, h, w)))
    else:
        return np.zeros((h, w), dtype=np.uint8)
    return (binary[:, :, 0] if binary.ndim == 3 else binary).astype(np.uint8)


def _augment(img: Image.Image, mask: Image.Image):
    """Synchronised augmentation — applied identically to image and mask."""
    if random.random() > 0.5:          # horizontal flip
        img  = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.8:          # vertical flip
        img  = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() > 0.8:          # random 90° rotation
        k    = random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
        img  = img.transpose(k)
        mask = mask.transpose(k)
    return img, mask


# =============================================================================
# DataLoader helpers
# =============================================================================

def _make_loader(dataset, batch_size: int, shuffle: bool, distributed: bool):
    if dataset is None:
        return None
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        num_workers=0, # config.get('num_workers', 4),
        pin_memory=False, # torch.cuda.is_available(),
        sampler=sampler,
        drop_last=shuffle,
    )


def make_data_loaders(distributed: bool):
    """
    Build train / val loaders.

    num_classes and class_names are read from the JSON annotation file itself
    (via COCOSegDataset) and stored back into config so that prepare_training()
    and evaluate() can consume them without touching the YAML classes list.
    """
    cc         = config['coco']
    batch_size = config.get('batch_size', 2)
    common     = dict(
        coco_root=cc['coco_root'],
        ignore_bg=config.get('ignore_bg', False),
        inp_size=config.get('inp_size', 1024),
    )

    train_ds = COCOSegDataset(split='train', augment=config.get('augment', True), **common)
    val_ds   = COCOSegDataset(split='val',   augment=False,                        **common)

    # Store num_classes and class_names derived from the JSON for downstream use
    config['_num_classes'] = train_ds.num_classes
    config['_class_names'] = train_ds.class_names

    if is_main:
        log(f'train: {len(train_ds)} samples  |  val: {len(val_ds)} samples')
        s = train_ds[0]
        for k, v in s.items():
            log(f'  {k}: shape={tuple(v.shape)}  dtype={v.dtype}')

    return (
        _make_loader(train_ds, batch_size, shuffle=True,  distributed=distributed),
        _make_loader(val_ds,   batch_size, shuffle=False, distributed=distributed),
    )


# =============================================================================
# Model
# =============================================================================

def prepare_training(device):
    # num_classes comes from the JSON annotation file (set by make_data_loaders)
    # = number of JSON categories + 1 (background).  E.g. 9 categories -> 10 classes.
    num_classes = config['_num_classes']
    config['model']['args']['num_classes'] = num_classes

    model = models.make(config['model'])

    # Load SAM2 backbone weights
    sam_path = config.get('sam_checkpoint')
    if sam_path and os.path.isfile(sam_path):
        ckpt = torch.load(sam_path, map_location='cpu')

        for k,v in ckpt["model"].items():
            if "patch_embed.proj.weight" in k:
                print(v.shape)

        sd   = ckpt.get('model', ckpt)
        miss, unex = model.load_state_dict(sd, strict=False)
        if is_main:
            log(f'SAM2 checkpoint loaded: {sam_path}')
            if miss:  log(f'  missing  : {len(miss)} (adapter layers — expected)')
            if unex:  log(f'  unexpected: {len(unex)}')
    elif is_main:
        log('WARNING: sam_checkpoint not found — training from scratch.')

    # Freeze image encoder; adapters + prompt generator stay trainable
    if config.get('freeze_encoder', True):
        for name, param in model.named_parameters():
            if 'image_encoder' in name and 'prompt_generator' not in name:
                param.requires_grad_(False)

    if is_main:
        total = sum(p.numel() for p in model.parameters())
        grad  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f'params: {utils.compute_num_params(model, text=True)} total  '
            f'{grad:,} trainable  {total - grad:,} frozen')

    # Optional resume
    resume = config.get('resume')
    epoch_start = config.get('start_epoch', 1)
    if resume and os.path.isfile(resume):
        model.load_state_dict(torch.load(resume, map_location='cpu'), strict=False)
        if is_main:
            log(f'Resumed from: {resume}  (start_epoch={epoch_start})')

    model = model.to(device)
    optimizer    = utils.make_optimizer(model.parameters(), config['optimizer'])
    lr_scheduler = CosineAnnealingLR(
        optimizer, T_max=config['epoch_max'], eta_min=config.get('lr_min', 1e-7)
    )
    return model, optimizer, epoch_start, lr_scheduler


# =============================================================================
# Training loop
# =============================================================================
def train_one_epoch(loader, model, optimizer, scaler, epoch, device, distributed):
    model.train()
    if distributed:
        loader.sampler.set_epoch(epoch)

    accum  = config.get('accum_steps', 1)
    pbar   = tqdm(total=len(loader), leave=False, desc='train') if is_main else None
    losses = []
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        model.set_input(batch['inp'], batch['gt'])
        model.optimize_parameters()

        last_step = (step + 1) % accum == 0 or (step + 1) == len(loader)
        if last_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        losses.append(model.loss_G.item())
        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()
    return mean(losses)

# =============================================================================
# Validation loop
# =============================================================================

@torch.no_grad()
def evaluate(val_loader, model, device):
    if val_loader is None:
        return None, None, None

    model.eval()
    classes    = config['_class_names']   # derived from JSON, not YAML
    ignore_bg  = config.get('ignore_bg', False)
    metric_seg = SegmentationMetric(config['_num_classes'], ignore_bg)
    pbar       = tqdm(total=len(val_loader), leave=False, desc='val') if is_main else None

    for batch in val_loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        pred   = torch.sigmoid(model.infer(batch['inp'])).cpu()   # (B, C, H, W)
        gt     = batch['gt'].cpu()                                  # (B, H, W)

        for i in range(pred.shape[0]):
            pred_idx = pred[i].argmax(dim=0).numpy().flatten()
            gt_idx   = gt[i].numpy().flatten()
            valid    = gt_idx != 255
            metric_seg.addBatch(pred_idx[valid], gt_idx[valid])

        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    oa        = np.around(metric_seg.overallAccuracy(), 4)
    mIoU, IoU = metric_seg.meanIntersectionOverUnion()
    mIoU      = np.around(mIoU, 4)
    IoU       = np.around(IoU,  4)
    p         = np.around(metric_seg.precision(), 4)
    r         = np.around(metric_seg.recall(),    4)
    f1        = np.around(2 * p * r / np.where((p + r) == 0, 1, p + r), 4)
    fwIoU     = np.around(metric_seg.Frequency_Weighted_Intersection_over_Union(), 4)
    normed_cm = np.around(
        metric_seg.confusionMatrix / (metric_seg.confusionMatrix.sum(axis=0) + 1e-8), 3
    )

    labels    = classes[1:] if ignore_bg else classes
    table     = PrettyTable(['Metric', 'Mean'] + list(labels))
    table.add_row(['IoU',       mIoU]           + IoU.tolist())
    table.add_row(['Precision', np.nanmean(p)]  + p.tolist())
    table.add_row(['Recall',    np.nanmean(r)]  + r.tolist())
    table.add_row(['F1',        np.nanmean(f1)] + f1.tolist())
    table.add_row(['OA',        oa]             + [' '] * len(labels))
    table.add_row(['FwIoU',     fwIoU]          + [' '] * len(labels))

    return float(mIoU), str(table), normed_cm


# =============================================================================
# Checkpoint + early stopping
# =============================================================================

def save_checkpoint(model, save_path: str, name: str):
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f'model_epoch_{name}.pth')
    sd   = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(sd, path)
    if is_main:
        log(f'Checkpoint -> {path}')


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = None
        self.wait      = 0

    def step(self, value: float) -> bool:
        """Return True when training should stop."""
        if self.best is None or value > self.best + self.min_delta:
            self.best = value
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience


# =============================================================================
# Main
# =============================================================================

def main(config_: dict, save_path: str):
    global config, log, writer, local_rank, is_main
    config = config_
    config['use_amp'] = False

    # ── Distributed setup ─────────────────────────────────────────────────────
    distributed = 'LOCAL_RANK' in os.environ
    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        is_main = (local_rank == 0)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main:
        log, writer = utils.set_save_path(save_path, remove=False)
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        log(f'Device: {device}  |  distributed: {distributed}')
    else:
        log = lambda *a, **kw: None

    train_loader, val_loader = make_data_loaders(distributed)
    model, optimizer, epoch_start, lr_scheduler = prepare_training(device)
    model.optimizer = optimizer

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    use_amp = config.get('use_amp', False) and torch.cuda.is_available()
    scaler  = GradScaler(enabled=use_amp)

    es      = config.get('early_stopping', {})
    stopper = EarlyStopping(
        patience=es.get('patience', 20),
        min_delta=es.get('min_delta', 0.001),
    ) if es.get('patience') else None

    epoch_max  = config['epoch_max']
    epoch_val  = config.get('epoch_val', 1)
    epoch_save = config.get('epoch_save', 5)
    best_mIoU  = -1.0
    timer      = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t0         = timer.t()
        train_loss = train_one_epoch(
            train_loader, model, optimizer, scaler, epoch, device, distributed
        )
        lr_scheduler.step()

        if is_main:
            lr_now = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr',         lr_now,     epoch)
            writer.add_scalar('loss/train', train_loss, epoch)
            log_parts = [f'epoch {epoch}/{epoch_max}',
                         f'lr={lr_now:.2e}',
                         f'train_loss={train_loss:.4f}']

        if is_main and epoch % epoch_save == 0:
            save_checkpoint(model, save_path, 'last')

        if epoch_val and epoch % epoch_val == 0:
            _m = model.module if hasattr(model, 'module') else model
            mIoU, table_str, confusion = evaluate(val_loader, _m, device)

            if is_main and mIoU is not None:
                writer.add_scalar('val/mIoU', mIoU, epoch)
                log_parts.append(f'val_mIoU={mIoU:.4f}')

                if mIoU > best_mIoU:
                    best_mIoU = mIoU
                    save_checkpoint(model, save_path, 'best')
                    log_parts.append('← best')

                prog  = (epoch - epoch_start + 1) / max(epoch_max - epoch_start + 1, 1)
                t_ep  = utils.time_text(timer.t() - t0)
                t_tot = utils.time_text(timer.t() / prog) if prog > 0 else '?'
                log_parts += [t_ep, f'{utils.time_text(timer.t())}/{t_tot}']
                log_parts.append('\n' + table_str)
                if confusion is not None:
                    log_parts.append('Confusion Matrix:\n' + str(confusion))

                log(' | '.join(log_parts))
                writer.flush()

                if stopper and stopper.step(mIoU):
                    log(f'Early stopping at epoch {epoch}  (best mIoU={best_mIoU:.4f})')
                    break
            elif is_main:
                log(' | '.join(log_parts))

    if is_main:
        save_checkpoint(model, save_path, 'last')
        log(f'Training complete. Best val mIoU={best_mIoU:.4f}')
        writer.close()

    if distributed:
        dist.destroy_process_group()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',        default='configs/train.yaml')
    parser.add_argument('--name',          default=None)
    parser.add_argument('--tag',           default=None)
    parser.add_argument('--prepare-only',  action='store_true',
                        help='Restructure dataset then exit (no training).')
    parser.add_argument('--force-prepare', action='store_true',
                        help='Force re-run of dataset preparation.')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    config = cfg   # needed by _categories_from_cfg() during preparation

    if cfg.get('split_preparation'):
        prepare_coco_splits(cfg, force=args.force_prepare)
    elif args.prepare_only:
        print('No split_preparation block in config — nothing to prepare.')

    if args.prepare_only:
        raise SystemExit(0)

    run_name  = args.name or ('_' + os.path.splitext(os.path.basename(args.config))[0])
    if args.tag:
        run_name += '_' + args.tag

    main(cfg, os.path.join('./save', run_name))