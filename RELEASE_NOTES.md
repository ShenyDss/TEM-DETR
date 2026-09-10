# Repository packaging notes

The reference YAML is `configs/tem/tem_anomaly_only_mvtec.yml`.
Its hyperparameters are retained; dataset locations use portable relative paths.
TEM modules are imported from `src/zoo/tem_modules`, with no dependency on an
editable installation of the earlier `tem_detr` package. Model parameter names
are preserved by the relocation.

Changes beyond relocation:

- Added paired COCO validation/configuration helper and nine TEM backbone configurations.
- Postprocessing caps the requested number of predictions by available
  query/class combinations, allowing small-class datasets with Top-K selection.
- The CLI AMP flag defaults to unset, so YAML `use_amp` is respected unless
  explicitly overridden. The earlier CLI default False could mask YAML True.
  To reproduce a historical run that used AMP off, explicitly pass
  `-u use_amp=False`.

Removed from the export: classification, VOC, YOLO/CSP backbones, RT-DETRv1
decoder/criterion, parameter conversion, visualization, the legacy standalone
TEM graph/factory, local results, weights, datasets and temporary artifacts.
The original source workspace is untouched.

Validation performed on CPU with the versions in requirements-tested.txt:

- Seven TEM backbone configurations: instantiate without downloading weights,
  single-image forward, finite logits and [1,20,4] output boxes at 128px.
- R50: one synthetic paired training epoch, all TEM loss branches,
  backward/optimizer step, COCO bbox evaluation, best/last checkpoint saving.
- Dataset validation/config generation and training CLI help.

The synthetic test is a runtime check, not an accuracy reproduction.
Real-data training and fresh-environment installation were not rerun for this export.

## Transformer backbone extension

Added torchvision Swin-T and ViT-B/16 with ImageNet-1K V1 pretraining options.
ViT uses layers 4/8/12 with a learned projection and resampling pyramid;
positional embeddings are interpolated to the runtime patch grid.
Both passed CPU paired TEM forward/backward with finite, nonzero backbone
gradients at 128px and single-image inference at 448px (20 output queries).
Checks used random initialization; pretrained downloads and training accuracy
were not verified. ResNet-101-vd remains available via tem_r101.yml.
