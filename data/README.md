# Dataset

LLVIP is not stored in git.

1. Read the licence: https://github.com/bupt-ai-cz/LLVIP/blob/main/Term%20of%20Use%20and%20License.md
2. Download the aligned release listed in https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md
3. Arrange the files as:

```text
data/llvip/
├── visible/train/
├── visible/test/
├── infrared/train/
├── infrared/test/
└── Annotations/
```

Visible and infrared files that form a pair share a filename. Annotations are VOC XML and are used only by the optional detection experiment.

Then:

```text
python scripts/prepare_dataset.py --root data/llvip
```

That command writes `data/splits/` (test, validation, nested training lists, gallery). Those lists are the protocol. Do not edit them after training has started.
