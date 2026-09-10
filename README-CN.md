# TEM-DETR 中文使用指南

[仓库首页](README.md)

基于 RT-DETRv2 的正常模板引导工业缺陷检测。本仓库整理
`configs/tem/tem_anomaly_only_mvtec.yml`，训练入口为 `tools/train.py`。
代码中包含 CVMA、LQSD、NQG 和差异 Top-K；训练使用空间对齐图像对，模型推理使用异常图像。

## 1. 安装

建议 Python 3.11。在仓库根目录执行：

```bash
conda create -n tem-detr python=3.11 -y
conda activate tem-detr
```

先按 https://pytorch.org/get-started/locally/ 安装与 GPU 驱动匹配的
PyTorch 和 torchvision，再执行：

```bash
python -m pip install -r requirements.txt
python tools/train.py --help
```

首次构建模型会下载上游 ImageNet 主干预训练权重，请确保能够访问 GitHub。
模型训练检查点不包含在仓库中。AMP 训练需要 CUDA；CPU 检查请传
`-d cpu -u use_amp=False`。依赖的本地验证版本见 `requirements-tested.txt`。

## 2. 准备数据

准备如下目录（可放在任意磁盘；下面的 `data` 是默认相对位置）：

```text
data/
  images/                 # 缺陷图像，可以有子目录
  normal/                 # 对应正常模板
  annotations/
    train.json
    val.json
```

标注使用 COCO bbox 格式。每个 `images` 项增加 `normal_file_name`，相对
`normal/`；`file_name` 相对 `images/`。示例（需替换成真实图像及完整标注）：

```json
{
  "images": [{"id": 1, "file_name": "bottle/001.png", "normal_file_name": "bottle/good/001.png", "width": 1024, "height": 1024}],
  "categories": [{"id": 1, "name": "bottle"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 200, 40, 30], "area": 1200, "iscrowd": 0}]
}
```

框为原图像素坐标 `[x,y,width,height]`，不做预归一化。训练和验证类别 ID/名称
必须一致。图像对必须同尺寸且内容空间对齐。不同划分应避免同源图像泄漏。
旧数据也支持文件名规则 `product__index__*.png` 对应
`normal/product/train/good/index.png`；下面的准备工具要求显式
`normal_file_name`，便于检查配对。

类别 ID 直接用作模型标签（`remap_mscoco_category: False`）。因此 ID 为
1–15 时 `num_classes=16`，0 是未使用输出槽位，不是背景类；ID 为 0–14 时
为 15。准备工具自动设置为 `max(category_id)+1`。类别名称以标注定义为准，
产品类别不等于缺陷类型类别。

数据在默认位置时：

```bash
python tools/prepare_config.py --data-root ./data --output configs/local.yml
```

外部数据目录则将 `./data` 替换为自己的目录，例如 `"/path/to/paired_dataset"`
（Windows 可使用 `"D:/datasets/paired_dataset"`）。工具检查所有图像对、类别、
标注引用和尺寸，然后生成配置；生成的本地配置无需上传 GitHub。

## 3. 训练、验证和续训

以下命令在仓库根目录执行，PowerShell/bash 均可直接复制单行命令：

```bash
python -u tools/train.py -c configs/local.yml -d cuda --seed 42 --output-dir outputs/tem_r50 -u epoches=50
```

复现指定源 YAML 的默认训练长度为 **72 epochs**；50 是上面命令的显式覆盖。
其他默认值：448×448、总 batch=2、workers=0、300 个 decoder queries、
Top-K=20（从 epoch 0 开始），CVMA mask=20%，LQSD temperature=2。
CVMA/LQSD/NQG 损失权重分别为 0.1/0.1/0.05（模型构造默认值）。
AdamW 主学习率 1e-4，ResNet 主干分组 1e-5，warmup 2000 steps，EMA/AMP 开启。

显存不足可在 `-u` 后追加 `train_dataloader.total_batch_size=1`
和 `val_dataloader.total_batch_size=1`。不要同时设置 `batch_size` 和
`total_batch_size`。更换尺寸时需同时更新 `eval_spatial_size`、两套 Resize 和
collate scales。

```bash
python -u tools/train.py -c configs/local.yml -d cuda --test-only -r outputs/tem_r50/best.pth
python -u tools/train.py -c configs/local.yml -d cuda -r outputs/tem_r50/last.pth --output-dir outputs/tem_r50 -u epoches=50
```

检查点实际文件名以输出目录为准。训练按 batch 打印进度，默认每 100 batch，
每个 epoch 验证一次并打印 COCO bbox 指标（faster-coco-eval）。更频繁输出可追加
`print_freq=20`。输出包含检查点、JSON 行日志和 TensorBoard 文件。
验证数据加载器目前仍要求正常模板，即使模型的 eval 前向不使用模板。

## 4. 主干配置

| TEM 配置 | 主干 |
|---|---|
| configs/tem/tem_anomaly_only_mvtec.yml | ResNet-50-vd，原始参考配置 |
| configs/tem/tem_r18.yml | ResNet-18-vd |
| configs/tem/tem_r34.yml | ResNet-34-vd |
| configs/tem/tem_r101.yml | ResNet-101-vd |
| configs/tem/tem_swin_t.yml | Swin-T |
| configs/tem/tem_vit_b16.yml | ViT-B/16（含多尺度检测适配器） |
| configs/tem/tem_hgnetv2_l.yml | HGNetv2-L |
| configs/tem/tem_hgnetv2_x.yml | HGNetv2-X |
| configs/tem/tem_hgnetv2_h.yml | HGNetv2-H |

例如更换 ResNet-18：

```bash
python tools/prepare_config.py --data-root ./data --config configs/tem/tem_r18.yml --output configs/local_r18.yml
python -u tools/train.py -c configs/local_r18.yml -d cuda --seed 42 --output-dir outputs/tem_r18 -u epoches=50
```

新增主干配置沿用 TEM-R50 的数据协议与优化器；不是各主干的已调优实验结果。
原上游主干配置保留在 `configs/rtdetrv2/`，供参考。

Swin-T 和 ViT-B/16 使用 torchvision，不需要 timm。首次运行默认下载
torchvision 的 ImageNet-1K V1 权重。两种主干内部对 [0,1] RGB 输入执行
ImageNet 标准化，数据变换中不要再重复 Normalize。
Swin-T 返回第 2/3/4 阶段的 stride 8/16/32 特征；ViT-B/16 取第 4/8/12 层
patch token，通过 1×1 投影、上采样/池化构造相同尺度的特征金字塔。
ViT 使用插值位置编码支持 448 输入，输入高宽必须能被 32 整除。
该适配方案是本仓库的实现选择，未宣称与未提供的论文主干实现细节完全一致。

```bash
python tools/prepare_config.py --data-root ./data --config configs/tem/tem_swin_t.yml --output configs/local_swin.yml
python -u tools/train.py -c configs/local_swin.yml -d cuda --seed 42 --output-dir outputs/tem_swin_t -u epoches=50
python tools/prepare_config.py --data-root ./data --config configs/tem/tem_vit_b16.yml --output configs/local_vit.yml
python -u tools/train.py -c configs/local_vit.yml -d cuda --seed 42 --output-dir outputs/tem_vit_b16 -u epoches=50
```

两种新主干的配对训练梯度和 448px 推理检查：`python tools/check_transformers.py`。

## 5. 实现与复现边界

`src/zoo/tem_rtdetr.py` 为集成模型，`src/zoo/tem_modules/` 为 TEM 模块；
`src/data/dataset/paired_coco_dataset.py` 为配对数据接口；
`src/solver/` 为训练与 COCO 验证。原独立 TEM 模型/工厂、可视化、历史配置、
实验结果、缓存、数据和权重未包含。

本次整理保留当前算法行为：训练 Top-K 比较两分支 decoder query；推理时
NQG 从单图 memory 生成正常 query 再筛选。训练仅对最后一层检测输出做筛选，
aux/encoder 输出保留全部 query；paired decode 路径未调用上游 denoising 分支。
本仓库提供运行复现入口；复现某一论文数值还需要相同数据划分、预训练权重、
随机种子和训练配置。未将主干 smoke test 作为精度实验。

验证整理版本的训练/验证链路：

```bash
python tools/smoke_test.py
```
