"""Exercise all TEM backbone graphs, then one CPU train/eval epoch on dummy COCO pairs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-backbones', action='store_true')
    args = parser.parse_args()
    import torch
    from src.core import YAMLConfig
    torch.set_num_threads(2)
    if not args.skip_backbones:
        for path in sorted((ROOT / 'configs/tem').glob('*.yml')):
            cfg = YAMLConfig(str(path), PResNet={'pretrained': False}, HGNetv2={'pretrained': False},
                             SwinTBackbone={'pretrained': False}, ViTB16Backbone={'pretrained': False},
                             eval_spatial_size=[128, 128])
            model = cfg.model.eval()
            with torch.no_grad():
                pred = model(torch.rand(1, 3, 128, 128))
            assert pred['pred_boxes'].shape == (1, 20, 4)
            assert torch.isfinite(pred['pred_logits']).all()
            print('BACKBONE PASS:', path.name, flush=True)
            del model, cfg, pred
    with tempfile.TemporaryDirectory(prefix='tem-smoke-') as directory:
        root = Path(directory)
        for folder in ('images', 'normal', 'annotations'):
            (root / folder).mkdir()
        records, annotations = [], []
        for i in (1, 2):
            normal = Image.new('RGB', (64, 64), (100 + i, 120, 140))
            anomaly = normal.copy()
            ImageDraw.Draw(anomaly).rectangle((16, 16, 31, 31), fill=(220, 20, 20))
            normal.save(root / 'normal' / f'{i}.png')
            anomaly.save(root / 'images' / f'{i}.png')
            records.append(dict(id=i, file_name=f'{i}.png', normal_file_name=f'{i}.png', width=64, height=64))
            annotations.append(dict(id=i, image_id=i, category_id=1, bbox=[16, 16, 16, 16], area=256, iscrowd=0))
        doc = dict(images=records, annotations=annotations, categories=[dict(id=1, name='synthetic_defect')])
        for split in ('train', 'val'):
            (root / 'annotations' / f'{split}.json').write_text(json.dumps(doc), encoding='utf-8')
        local = root / 'smoke.yml'
        subprocess.run([sys.executable, '-B', str(ROOT / 'tools/prepare_config.py'), '--data-root', str(root), '--output', str(local)], check=True, cwd=ROOT)
        config = yaml.safe_load(local.read_text())
        config.update(epoches=1, device='cpu', use_amp=False, use_ema=False, eval_spatial_size=[128, 128],
                      output_dir=str(root / 'output'), print_freq=1)
        config['PResNet']['pretrained'] = False
        for loader in ('train_dataloader', 'val_dataloader'):
            config[loader]['dataset']['transforms']['ops'][0]['size'] = [128, 128]
            config[loader]['collate_fn']['scales'] = [128]
        local.write_text(yaml.safe_dump(config, sort_keys=False))
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-B', '-u', str(ROOT / 'tools/train.py'), '-c', str(local), '-d', 'cpu', '--seed', '42']
        subprocess.run(command, check=True, cwd=ROOT, env=env)
        log = json.loads((root / 'output/log.txt').read_text().splitlines()[-1])
        assert log['epoch'] == 0 and 'test_coco_eval_bbox' in log
        assert all('train_loss_' + key in log for key in ['cvma_cos', 'lqsd_kl', 'nqg_cos'])
        assert (root / 'output/best.pth').is_file()
        assert (root / 'output/last.pth').is_file()
        subprocess.run(command + ['--test-only', '-r', str(root / 'output/best.pth')], check=True, cwd=ROOT, env=env)
        print('PASS: paired data, forward/backward, TEM losses, COCO evaluation and checkpoints', flush=True)


if __name__ == '__main__':
    main()
