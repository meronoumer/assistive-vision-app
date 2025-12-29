# Assistive-Vision training helper

This repository includes a training script to fine-tune MobileNetV2 for a small
multi-label dataset derived from COCO (or an ImageNet subset). The goal is to
export a TensorFlow Lite model suitable for mobile inference.

Quickstart (example):

1. Install dependencies (preferably in a virtualenv):

```bash
pip install -r requirements.txt
```

2. Run a short training session (this will download TFDS datasets if needed):

```bash
python train_mobilenet_transfer.py --dataset coco --epochs 3 --batch_size 32 --output_dir ./models
```

3. The trained SavedModel and TFLite file will be in `./models`.

Notes:
- Full COCO or ImageNet training requires substantial disk and compute resources; use
  the `--limit_per_class` and split slices to experiment locally.
- To get mobile latency <200ms, convert with `--tflite_quant float16` or int8 and test
  on target hardware.
