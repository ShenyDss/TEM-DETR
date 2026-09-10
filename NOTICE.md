# Upstream attribution

This repository packages the TEM-DETR integration on top of RT-DETRv2.
Upstream: https://github.com/lyuwenyu/RT-DETR (rtdetrv2_pytorch).
Upstream copyright notices and Apache-2.0 LICENSE are retained.
Local modifications include paired COCO loading/transforms, paired training,
CVMA, LQSD, NQG and query selection. The source snapshot did not establish an
upstream commit identifier; none is inferred here.

TEM-specific files were migrated from the supplied research implementation.
Dataset images, annotations and pretrained weights are not distributed here;
their original licenses and citations apply separately.
