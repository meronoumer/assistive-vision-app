"""
test_dataset.py

Loads sample images from TFDS (COCO or ImageNet subset), displays a grid with labels,
prints dataset statistics, verifies image dimensions and dtypes, and checks for
corrupted images or missing labels.

Example:
  python test_dataset.py --dataset coco --num_samples 10
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except Exception as e:
    print("Missing TensorFlow or TFDS. Install requirements.txt and try again.")
    raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "imagenet_subset"], default="cifar10")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--check_samples", type=int, default=1000,
                   help="Number of examples to scan for corrupted images/missing labels (0 to skip)")
    return p.parse_args()


def get_builder(dataset_key):
    if dataset_key == "cifar10":
        return tfds.builder("cifar10")
    if dataset_key == "imagenet_subset":
        return tfds.builder("imagenet2012")
    raise ValueError("Unsupported dataset")


def load_samples(dataset_key, num_samples=10):
    if dataset_key == "cifar10":
        ds = tfds.load("cifar10", split="train", shuffle_files=True, as_supervised=True)
    else:
        ds = tfds.load("imagenet2012", split="train[:1%]", shuffle_files=True)

    samples = []
    # tfds.as_numpy for as_supervised yields tuples (img, label)
    for ex in tfds.as_numpy(ds.take(num_samples)):
        if dataset_key == "cifar10":
            img, lbl = ex
            labels = [int(lbl)] if lbl is not None else []
        else:
            img = ex.get("image")
            lbl = ex.get("label")
            labels = [int(lbl)] if lbl is not None else []
        samples.append((img, labels))
    return samples


def show_grid(samples, max_cols=5):
    n = len(samples)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for i, (img, labels) in enumerate(samples):
        ax = axes[i]
        img_disp = img.astype(np.uint8) if img.dtype != np.uint8 else img
        ax.imshow(img_disp)
        title = ", ".join([str(l) for l in labels]) if labels else "(no labels)"
        ax.set_title(title[:60])
        ax.axis("off")

    # hide remaining axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def print_dataset_stats(builder, dataset_key):
    info = builder.info
    print("Dataset name:", info.name)
    # splits
    try:
        for sname, splitinfo in info.splits.items():
            print(f"Split {sname}: {splitinfo.num_examples} examples")
    except Exception:
        print("Could not read split counts; the dataset may not be prepared locally.")

    # classes
    try:
        if dataset_key == "cifar10":
            print("Number of classes: 10 (CIFAR-10)")
        else:
            print("Number of classes:", info.features["label"].num_classes)
    except Exception:
        print("Could not determine class count from builder info.")


def verify_samples(samples):
    print("Verifying image dimensions and dtypes for samples:")
    shapes = [s[0].shape for s in samples]
    dtypes = [s[0].dtype for s in samples]
    print("Shapes:", shapes)
    print("DTypes:", dtypes)
    # check consistency
    if len(set(shapes)) == 1:
        print("All sample images have consistent shape.")
    else:
        print("Inconsistent image shapes detected.")


def scan_for_issues(dataset_key, max_check=1000):
    if max_check <= 0:
        print("Skipping scan for corrupted images and missing labels.")
        return

    print(f"Scanning up to {max_check} examples for corrupted images or missing labels...")
    if dataset_key == "cifar10":
        ds = tfds.load("cifar10", split="train", as_supervised=True)
    else:
        ds = tfds.load("imagenet2012", split="train[:1%]")

    corrupted = 0
    missing_labels = 0
    total = 0
    for ex in tfds.as_numpy(ds.take(max_check)):
        total += 1
        if dataset_key == "cifar10":
            img, _ = ex
        else:
            img = ex.get("image")
        if img is None:
            corrupted += 1
            continue
        try:
            # simple checks
            if getattr(img, "size", None) is None or img.size == 0:
                corrupted += 1
        except Exception:
            corrupted += 1

        # labels
        # For CIFAR-10 the label is the second element when as_supervised=True
        if dataset_key == "cifar10":
            try:
                _, label = ex
                if label is None:
                    missing_labels += 1
            except Exception:
                missing_labels += 1
        else:
            lbl = ex.get("label")
            if lbl is None:
                missing_labels += 1

    print(f"Scanned {total} examples — corrupted: {corrupted}, missing labels: {missing_labels}")


def main():
    args = parse_args()
    builder = get_builder(args.dataset)
    try:
        builder.download_and_prepare()
    except Exception:
        print("Dataset may already be prepared or automatic download not possible.")

    print_dataset_stats(builder, args.dataset)

    samples = load_samples(args.dataset, num_samples=args.num_samples)
    if not samples:
        print("No samples found. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(samples)} samples. Displaying grid...")
    verify_samples(samples)
    show_grid(samples)

    scan_for_issues(args.dataset, max_check=args.check_samples)


if __name__ == "__main__":
    main()
