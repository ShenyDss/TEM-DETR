"""Validate a portable paired COCO dataset and write a runnable TEM config."""
import argparse
import json
from pathlib import Path
import sys
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate(root):
    categories = None
    maximum = -1
    for split in ('train', 'val'):
        document = json.loads((root / 'annotations' / f'{split}.json').read_text(encoding='utf-8'))
        current = {c['id']: c['name'] for c in document['categories']}
        if not current or any(type(k) is not int or k < 0 for k in current):
            raise ValueError('Category IDs must be nonnegative integers')
        if categories is not None and current != categories:
            raise ValueError('Train/val category definitions differ')
        categories = current
        ids = set()
        for record in document['images']:
            if record['id'] in ids:
                raise ValueError('Duplicate image ID')
            ids.add(record['id'])
            paths = []
            for folder, field in [('images', 'file_name'), ('normal', 'normal_file_name')]:
                relative = Path(record[field])
                if relative.is_absolute() or '..' in relative.parts or ':' in str(relative):
                    raise ValueError(f'{field} must be a portable relative path')
                paths.append(root / folder / relative)
            with Image.open(paths[0]) as anomaly, Image.open(paths[1]) as normal:
                if anomaly.size != normal.size or anomaly.size != (record['width'], record['height']):
                    raise ValueError(f'Image sizes disagree: {record["file_name"]}')
        if not ids:
            raise ValueError(f'{split} is empty')
        annotation_ids = set()
        for a in document['annotations']:
            if a['id'] in annotation_ids or a['image_id'] not in ids or a['category_id'] not in current:
                raise ValueError('Invalid annotation ID/image/category reference')
            annotation_ids.add(a['id'])
            if len(a['bbox']) != 4 or a['bbox'][2] <= 0 or a['bbox'][3] <= 0:
                raise ValueError('COCO bbox must be [x,y,width,height] with positive size')
            if 'area' not in a or 'iscrowd' not in a:
                raise ValueError('Annotations require area and iscrowd')
        maximum = max(maximum, max(current))
        print(f'{split}: {len(ids)} images, {len(document["annotations"])} boxes, {len(current)} categories')
    return maximum + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/tem/tem_anomaly_only_mvtec.yml')
    parser.add_argument('--output', type=Path, default=ROOT / 'configs/local.yml')
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    num_classes = validate(root)
    from src.core.yaml_utils import load_config
    cfg = load_config(str(args.config.resolve()), {})
    cfg.pop('__include__', None)
    cfg['num_classes'] = num_classes
    for split, loader in [('train', 'train_dataloader'), ('val', 'val_dataloader')]:
        dataset = cfg[loader]['dataset']
        dataset.update(img_folder=(root / 'images').as_posix(),
                       ann_file=(root / 'annotations' / f'{split}.json').as_posix(),
                       normal_root=(root / 'normal').as_posix())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    print(f'Wrote {args.output}; num_classes={num_classes} (max category ID + 1)')


if __name__ == '__main__':
    main()
