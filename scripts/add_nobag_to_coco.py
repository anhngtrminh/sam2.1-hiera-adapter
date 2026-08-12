#!/usr/bin/env python3
"""
Merge a folder of no-bag images into an existing COCO annotation JSON.

No-bag images are added to `images` with **zero** annotations so the
training pipeline can learn "predict nothing when there is no bag".

Typical layout (matches configs/train.yaml):
    data/
      train/                 # positive (bag) images
      annotations.json       # existing COCO metadata
    nobag_images/            # separate folder of no-bag photos

Example:
    python scripts/add_nobag_to_coco.py \\
        --ann data/annotations.json \\
        --nobag-dir path/to/nobag_images \\
        --images-dir data/train \\
        --out-ann data/annotations_with_nobag.json

Then re-prepare splits:
    python finetune.py --config configs/train.yaml --prepare-only --force-prepare

Update train.yaml source_ann_file to the new JSON before preparing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def _list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def _unique_name(stem: str, ext: str, used: set[str], prefix: str) -> str:
    """Return a filename that does not collide with `used`."""
    candidate = f'{prefix}{stem}{ext}'
    if candidate not in used:
        return candidate
    i = 1
    while True:
        candidate = f'{prefix}{stem}_{i}{ext}'
        if candidate not in used:
            return candidate
        i += 1


def merge_nobag(
    ann_path: Path,
    nobag_dir: Path,
    images_dir: Path,
    out_ann: Path,
    prefix: str = 'nobag_',
    copy_images: bool = True,
    dry_run: bool = False,
) -> dict:
    with open(ann_path, encoding='utf-8') as f:
        coco = json.load(f)

    if 'images' not in coco or 'annotations' not in coco:
        raise ValueError(f'{ann_path} is not a COCO dict (need images + annotations)')

    coco.setdefault('categories', [])
    coco.setdefault('info', {})
    coco.setdefault('licenses', [])

    existing_names = {im['file_name'] for im in coco['images']}
    next_img_id = max((im['id'] for im in coco['images']), default=0) + 1

    nobag_files = _list_images(nobag_dir)
    if not nobag_files:
        raise RuntimeError(f'No images found in {nobag_dir}')

    images_dir.mkdir(parents=True, exist_ok=True)

    added = []
    skipped_dup_content = 0

    for src in nobag_files:
        # Skip if this exact basename already exists as a no-bag entry
        # (re-running the script should be idempotent for same names).
        dest_name = _unique_name(src.stem, src.suffix.lower(), existing_names, prefix)

        # If the unprefixed or prefixed name already points at an image with
        # zero annotations, treat as already merged.
        already = None
        for name in (src.name, f'{prefix}{src.name}', dest_name):
            if name in existing_names:
                already = name
                break
        if already is not None:
            # Only skip when that image truly has no annotations
            img_id = next(
                im['id'] for im in coco['images'] if im['file_name'] == already
            )
            has_ann = any(a['image_id'] == img_id for a in coco['annotations'])
            if not has_ann:
                skipped_dup_content += 1
                continue

        try:
            with Image.open(src) as im:
                w, h = im.size
        except Exception as e:
            print(f'[warn] skip unreadable {src.name}: {e}')
            continue

        dest_path = images_dir / dest_name
        if copy_images and not dry_run:
            if not dest_path.exists():
                shutil.copy2(src, dest_path)

        entry = {
            'id': next_img_id,
            'file_name': dest_name,
            'width': w,
            'height': h,
        }
        # Optional tag for debugging / filtering later
        entry['nobag'] = True

        coco['images'].append(entry)
        existing_names.add(dest_name)
        added.append(entry)
        next_img_id += 1

    # Sanity: no-bag image_ids must not appear in annotations
    nobag_ids = {im['id'] for im in coco['images'] if im.get('nobag')}
    leaked = [a for a in coco['annotations'] if a['image_id'] in nobag_ids]
    if leaked:
        raise RuntimeError(
            f'Found {len(leaked)} annotations on no-bag images — aborting.'
        )

    n_pos = sum(1 for im in coco['images'] if not im.get('nobag'))
    n_neg = sum(1 for im in coco['images'] if im.get('nobag'))
    # Also count images with zero anns that were already empty (no nobag flag)
    ann_img_ids = {a['image_id'] for a in coco['annotations']}
    n_empty = sum(1 for im in coco['images'] if im['id'] not in ann_img_ids)

    summary = {
        'source_ann': str(ann_path),
        'nobag_dir': str(nobag_dir),
        'images_dir': str(images_dir),
        'out_ann': str(out_ann),
        'nobag_scanned': len(nobag_files),
        'nobag_added': len(added),
        'nobag_skipped_existing': skipped_dup_content,
        'total_images': len(coco['images']),
        'total_annotations': len(coco['annotations']),
        'positive_images_est': n_pos,
        'nobag_flagged': n_neg,
        'empty_annotation_images': n_empty,
    }

    if dry_run:
        print('[dry-run] No files written.')
    else:
        out_ann.parent.mkdir(parents=True, exist_ok=True)
        with open(out_ann, 'w', encoding='utf-8') as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)
        print(f'[ok] Wrote {out_ann}')

    print('--- summary ---')
    for k, v in summary.items():
        print(f'  {k}: {v}')
    if added:
        print('  examples added:')
        for im in added[:5]:
            print(f'    id={im["id"]}  {im["file_name"]}  ({im["width"]}x{im["height"]})')
        if len(added) > 5:
            print(f'    ... +{len(added) - 5} more')

    return summary


def main():
    p = argparse.ArgumentParser(
        description='Add no-bag images into a COCO metadata JSON for training.'
    )
    p.add_argument('--ann', required=True, type=Path,
                   help='Existing COCO annotations JSON (e.g. data/annotations.json)')
    p.add_argument('--nobag-dir', required=True, type=Path,
                   help='Folder containing no-bag images only')
    p.add_argument('--images-dir', required=True, type=Path,
                   help='Destination image folder used by training '
                        '(e.g. data/train)')
    p.add_argument('--out-ann', required=True, type=Path,
                   help='Output COCO JSON path '
                        '(e.g. data/annotations_with_nobag.json)')
    p.add_argument('--prefix', default='nobag_',
                   help='Filename prefix to avoid collisions (default: nobag_)')
    p.add_argument('--no-copy', action='store_true',
                   help='Only update JSON; do not copy images into --images-dir')
    p.add_argument('--dry-run', action='store_true',
                   help='Print summary without writing JSON or copying images')
    args = p.parse_args()

    if not args.ann.is_file():
        raise SystemExit(f'Annotation file not found: {args.ann}')
    if not args.nobag_dir.is_dir():
        raise SystemExit(f'No-bag folder not found: {args.nobag_dir}')

    merge_nobag(
        ann_path=args.ann,
        nobag_dir=args.nobag_dir,
        images_dir=args.images_dir,
        out_ann=args.out_ann,
        prefix=args.prefix,
        copy_images=not args.no_copy,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
