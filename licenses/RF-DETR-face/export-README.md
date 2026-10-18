---
license: apache-2.0
---

# rfdetr-onnx

RF-DETR exported to ONNX for use as ffrwd model pins. The network is
[Roboflow's RF-DETR](https://github.com/roboflow/rf-detr), Apache-2.0
for the Nano through Large sizes. This repo holds faithful exports of
four checkpoints, nothing more: two of Roboflow's own, one third-party
fine-tune for faces, and one fine-tune of our own for licence plates.

All are fp32, exported by the `rfdetr` pip package's own
`model.export(format="onnx")` at each variant's own training size.

| file | variant | checkpoint | input | outputs |
|---|---|---|---|---|
| rf-detr-large-2026.onnx | RF-DETR Large (2026), 33.9M params, 56.5 COCO AP | rf-detr-large-2026.pth | 1x3x704x704 | dets 1x300x4, labels 1x300x91 |
| rf-detr-seg-large.onnx | RF-DETR Seg Large, 36.2M params, 47.1 COCO AP | rf-detr-seg-large.pt | 1x3x504x504 | dets 1x200x4, labels 1x200x91, masks 1x200x126x126 |
| rf-detr-medium-face.onnx | RF-DETR Medium fine-tuned for faces, one class | rfdetr_medium_face.pth | 1x3x576x576 | dets 1x300x4, labels 1x300x1 |
| rf-detr-medium-plate.onnx | RF-DETR Medium fine-tuned for licence plates, one class | rfdetr_medium_plate.pth | 1x3x576x576 | dets 1x300x4, labels 1x300x2 |

## Hashes

| file | bytes | sha256 |
|---|---|---|
| rf-detr-large-2026.onnx | 128189032 | da7f1bc118c6071b48096149b5df01eb1caeb3bb60a262949ac0e41f7c96f705 |
| rf-detr-seg-large.onnx | 137933860 | e6154493b7ba38c285f416db812589267a7292a736fa8564577bcd27a9012e9e |
| rf-detr-medium-face.onnx | 127032882 | b594500a99c3a4287805b7afa15eb74ee559be937eb00315768612846f93cc45 |
| rf-detr-medium-plate.onnx | 126302012 | 2597bc5f0f71e2a6c8da3983367d774599d0e587b2e298238fd8fb71bd1af337 |

The checkpoints each export started from. The two official ones are as
the `rfdetr` package fetched them into `~/.roboflow/models`; the face
one is as downloaded from its repo; the plate one is the EMA weights
our training run wrote, kept by hash rather than here.

| checkpoint | bytes | sha256 |
|---|---|---|
| rf-detr-large-2026.pth | 135954129 | 0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38 |
| rf-detr-seg-large.pt | 145055866 | ca7b7c630ba22496067cc4f034c4e70c8e47fd7adcea04c3a49aa8c1755cbe6b |
| rfdetr_medium_face.pth | 133666287 | 7e2d5e5a4087ecf98a73685679ddb0d667dc6da512be29c0fba5469816ebfe06 |
| rfdetr_medium_plate.pth | 133776283 | 6d4c2821ede40269e83f694f474d137d0e67d066880ead09ec51abd0b864fcc4 |

The two COCO exports were made with `rfdetr==1.10.0`, the face and
plate exports with `rfdetr==1.10.1` (`scripts/export_face.py`,
`scripts/export_plate.py`), each with `torch`
from the package's own dependency set, opset 17, IR 8. The exporter
does not simplify the graph: `onnxsim` is a listed dependency it never
calls, so what is here is the graph as exported.

## Model I/O

The input is planar RGB in NCHW, the whole frame stretched to the
square with no aspect padding, scaled to 0..1 and then normalized with
ImageNet's mean (0.485, 0.456, 0.406) and standard deviation (0.229,
0.224, 0.225). The graphs do not bake that in: their first node reads
the input tensor directly.

`dets` is `cx, cy, w, h`, each a fraction of the picture. `labels` is
raw class logits, so a confidence is `sigmoid` of the logit. For the
two COCO exports the logits are in COCO's 91-slot category numbering -
slot 1 is person, 62 chair, 72 tv - and a query's class is the largest
of them. `masks` is raw mask logits per query over the whole square at
quarter resolution; a pixel is inside its instance where the logit
crosses zero. A DETR head returns no duplicates, so there is no NMS.

## The face export

`rf-detr-medium-face.onnx` is RF-DETR Medium (DINOv2 backbone) as
fine-tuned for single-class face detection by
[Herojayjay/RFDETR-Face-Detection](https://huggingface.co/Herojayjay/RFDETR-Face-Detection),
Apache-2.0, on a Kaggle face dataset of about 16,700 annotated images,
at 576x576. It is not Roboflow's checkpoint and not trained on
WIDER FACE; the WIDER FACE figures below are a check, not its
benchmark.

Its head is one logit wide: `labels` is `1x300x1`, index 0 is the
face, and there is no background slot. The checkpoint's own
`class_embed` is `(1, 256)`, so the export loads it with `num_classes`
left unset and lets the package align the head to the checkpoint.
Passing `num_classes=1` would tile the trained row into a second,
invented class.

Measured on 200 WIDER FACE validation images (3,378 annotated faces)
at a single 576 pass, matching at IoU 0.5, face size as the square
root of the box area on the original image:

| confidence | recall | precision | recall under 32 px | 32-96 px | over 96 px |
|---|---|---|---|---|---|
| 0.25 | 0.32 | 0.87 | 0.17 | 0.82 | 0.98 |
| 0.50 | 0.21 | 0.97 | 0.07 | 0.71 | 0.98 |

Small faces dominate that benchmark (2,649 of the 3,378 are under
32 px), and a single 576 pass cannot resolve them; the medium and
large columns are the figures that describe a face that fills more of
the frame.

## The plate export

`rf-detr-medium-plate.onnx` is RF-DETR Medium fine-tuned by us for
single-class licence plate detection, at 576x576, from Roboflow's
rf-detr-medium.pth with `rfdetr==1.10.1`, on Open Images V7's "Vehicle
registration plate" boxes (annotations CC BY 4.0, images CC BY 2.0)
with the group-of boxes dropped: 4,497 training images carrying 6,554
plates, 724 validation images carrying 985. Twenty epochs, batch 8 with
two steps of gradient accumulation, on one RTX 4090; validation mAP
50:95 peaked at 0.655 after the fourth epoch and the EMA weights of
that epoch are what was exported (`scripts/export_plate.py`). The
training data is what the model knows: photographs, mostly of parked
and passing cars, plates from many countries, few of them at night.

Its head is two logits wide: `rfdetr` 1.10.1 lays a one-class head out
as the class plus one spare slot, so `labels` is `1x300x2`, index 0 is
the plate, and index 1 was never trained toward anything and stays
quiet - on the 2,065 test images below not one query's spare logit
cleared 0.5. There is no background slot to subtract.

Measured on Open Images' own test split, 2,065 images holding 2,835
plates, at a single 576 pass, matching at IoU 0.5, plate size as the
box's width on the original image:

| confidence | recall | precision | recall under 32 px | 32-96 px | over 96 px |
|---|---|---|---|---|---|
| 0.25 | 0.87 | 0.91 | 0.47 | 0.96 | 0.98 |
| 0.50 | 0.80 | 0.98 | 0.23 | 0.90 | 0.97 |

Of the 2,835 plates, 580 are under 32 px wide, and a single 576 pass
cannot resolve most of them; the two right-hand columns describe a
plate that fills more of the frame.

## Not here

RF-DETR XLarge and 2XLarge, and their segmentation counterparts, are
under Roboflow's PML 1.0 rather than Apache-2.0, and are not exported
here. The size names changed in 2026: the pre-rename "large" (487 MB
fp32, 126M params) is today's XLarge, so check parameter counts and
file sizes rather than names.
