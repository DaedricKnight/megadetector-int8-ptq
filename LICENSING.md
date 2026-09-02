# Which MDV6 model can you put on a device?

*Not legal advice — this is a summary of what the artifacts themselves declare, with
the evidence for each claim so you can re-check it. Verified 2 September 2026 against
Pytorch-Wildlife 1.2.x and the Zenodo record
[15398270](https://zenodo.org/records/15398270).*

The `microsoft/MegaDetector`, `microsoft/Pytorch-Wildlife` and `microsoft/CameraTraps`
repositories are all **MIT**. That is the license of the *code*, and it is what most
people see. **It is not the license of the weights you download**, and for the default
models the two differ.

## The short answer

| Variant | Weights declare | Runtime code needed | Safe to embed in a product? |
|---|---|---|---|
| `MDV6-yolov9-c` / `-e` | **AGPL-3.0** | `ultralytics` (AGPL-3.0) | ❌ not without AGPL compliance |
| `MDV6-yolov10-c` / `-e` | **AGPL-3.0** | `ultralytics` (AGPL-3.0) | ❌ not without AGPL compliance |
| `MDV6-rtdetr-c` / `-e` | **AGPL-3.0** | `ultralytics` (AGPL-3.0) | ❌ not without AGPL compliance |
| **`MDV6-mit-yolov9-c` / `-e`** | no AGPL notice | vendored [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (**MIT**) | ✅ |
| **`MDV6-apa-rtdetr-c` / `-e`** | no AGPL notice | vendored [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR) (**Apache-2.0**) | ✅ |
| MDV5 (`md_v5a/b`) | — | YOLOv5 (AGPL-3.0) | ❌ same problem |

**If you are shipping a device or a hosted service, use `MDV6-mit-yolov9-*` or
`MDV6-apa-rtdetr-*`.** They exist, they are on the same Zenodo record, and they are
the reason the `mit`/`apa` infixes are in those filenames — but nothing in the docs
says so, and the infix is not self-explanatory.

## The evidence

**1. The Ultralytics-trained checkpoints declare AGPL-3.0 inside the file.** Not
inferred from the architecture's upstream repo — written into the checkpoint by the
training framework:

```python
>>> ck = torch.load("MDV6-yolov10-c.pt", map_location="cpu", weights_only=False)
>>> ck["license"], ck["version"], ck["train_args"]["imgsz"]
('AGPL-3.0 (https://ultralytics.com/license)', '8.3.27', 640)
```

All three carry it:

| File | `license` | `version` (Ultralytics) | `train_args.imgsz` |
|---|---|---|---|
| `MDV6-yolov9-c.pt` | `AGPL-3.0 (https://ultralytics.com/license)` | 8.1.45 | 640 |
| `MDV6-yolov10-c.pt` | `AGPL-3.0 (https://ultralytics.com/license)` | 8.3.27 | 640 |
| `MDV6-rtdetr-c.pt` | `AGPL-3.0 (https://ultralytics.com/license)` | 8.2.2 | 640 |

**2. The `mit` and `apa` checkpoints carry no such notice** and are not Ultralytics
artifacts at all — `MDV6-mit-yolov9-c.ckpt` is a PyTorch Lightning checkpoint
(`pytorch-lightning_version`, `state_dict`), `MDV6-apa-rtdetr-c.pth` is a native
RT-DETR checkpoint (`criterion`, `postprocessor`, `lr_warmup_scheduler`).

**3. The runtime dependency differs, and that is what actually binds you.**
`PytorchWildlife/models/detection/ultralytics_based/yolov8_base.py` does
`from ultralytics.models import yolo, rtdetr` — the `ultralytics` package is AGPL-3.0
(`importlib.metadata.metadata("ultralytics")["License"]` → `AGPL-3.0`). The MIT and
Apache paths import no such thing; their inference code is vendored into
Pytorch-Wildlife under `models/detection/yolo_mit/yolo/` and
`models/detection/rtdetr_apache/rtdetrv2_pytorch/`, and `grep -rl ultralytics` over
both directories returns nothing.

## Does exporting to ONNX get you out of it?

Partly, and it is worth being precise about which part.

**The runtime dependency: yes.** Once exported, inference needs only ONNX Runtime
(MIT) — the `ultralytics` package is not shipped and not linked. That removes the
clearest AGPL hook.

**The weights: unresolved.** The checkpoint asserts AGPL-3.0 over itself, and
Ultralytics' public position is that the license reaches models trained with their
software. Whether trained weights are a derivative work of the training code is not
settled law, and an ONNX export is a format conversion of those weights, not a clean
reimplementation. If your risk tolerance for that question is low, the `mit`/`apa`
variants remove it entirely rather than argue it.

This distinction matters most for exactly the audience MDV6's compact variants are
aimed at: whoever is flashing a model into a field unit is redistributing the weights,
which is where AGPL §13 and §5 bite hardest.

## Two things worth fixing upstream

**Vendored third-party code ships without its license text.** Neither
`models/detection/yolo_mit/yolo/` nor
`models/detection/rtdetr_apache/rtdetrv2_pytorch/` contains a LICENSE or NOTICE file
(`find … -iname '*licen*'` → empty). MIT and Apache-2.0 both require the license and
copyright notice to travel with redistributed copies, so adding those two files is a
small packaging fix that makes the permissive path defensibly permissive.

**The license-clean variants are undiscoverable.** They are selected by strings like
`MDV6-mit-yolov9-c` and live in separate classes (`MegaDetectorV6MIT`,
`MegaDetectorV6Apache`) with their own `IMAGE_SIZE`. A single table in the README —
essentially the one above — would save every downstream integrator the hour it takes
to work this out from checkpoint internals.

## Reproducing this table

```bash
pip install PytorchWildlife torch
python - <<'PY'
import torch, urllib.request
for f in ("MDV6-yolov9-c.pt", "MDV6-yolov10-c.pt", "MDV6-rtdetr-c.pt",
          "MDV6-mit-yolov9-c.ckpt", "MDV6-apa-rtdetr-c.pth"):
    urllib.request.urlretrieve(
        f"https://zenodo.org/records/15398270/files/{f}?download=1", f)
    ck = torch.load(f, map_location="cpu", weights_only=False)
    print(f, "->", ck.get("license"), "| ultralytics", ck.get("version"),
          "| imgsz", (ck.get("train_args") or {}).get("imgsz"))
PY
```
