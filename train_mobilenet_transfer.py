"""
train_mobilenet_transfer.py

Usage (example):
  python train_mobilenet_transfer.py --dataset coco --epochs 10 --batch_size 32 --output_dir models

Notes:
- Downloads a small subset (configurable) of COCO via tfds if available and builds a
  multi-label classification dataset for a list of common classes.
- Builds a MobileNetV2 transfer model, trains with augmentation, logs TensorBoard,
  checkpoints best weights, and converts the SavedModel to a TFLite file with
  optional post-training quantization. Measures simple CPU latency for the TFLite file.

This script targets producing a TFLite model for mobile deployment. It is written
to be conservative about dataset size so you can test quickly; for production-scale
training you should run on a GPU-enabled environment and use larger dataset splits.
"""
import argparse
import os
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


TARGET_COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "chair",
    "couch",
    "bed",
    "dining table",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "imagenet_subset", "local"], default="cifar10")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", default="./models")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--limit_per_class", type=int, default=2000,
                   help="Max images per class to keep memory/training small for quick experiments.")
    p.add_argument("--tflite_quant", choices=["none", "float16", "int8"], default="float16")
    return p.parse_args()


def prepare_cifar_dataset(image_size, batch_size, split="train[:80%]"):
    # CIFAR-10 via TFDS: images are (32,32,3), labels are integers 0-9
    ds = tfds.load("cifar10", split=split, shuffle_files=True, as_supervised=True)

    def _map(image, label):
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_augmenter(image_size):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.12, 0.12),
        tf.keras.layers.RandomTranslation(0.04, 0.04),
    ], name="data_augmentation")
    return data_augmentation


def build_model(num_classes, image_size, learning_rate=1e-4):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = inputs
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def representative_data_gen(ds, num_samples=100):
    # ds yields (image, label) batches
    it = iter(ds)
    count = 0
    for batch in it:
        images = batch[0]
        for i in range(images.shape[0]):
            img = images[i:i+1]
            yield [tf.cast(img, tf.float32)]
            count += 1
            if count >= num_samples:
                return


def convert_to_tflite(saved_model_dir, tflite_path, quant="float16", rep_ds=None):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quant == "none":
        tflite_model = converter.convert()
    elif quant == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
    elif quant == "int8":
        if rep_ds is None:
            raise ValueError("Representative dataset required for int8 quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: rep_ds
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
    else:
        raise ValueError("Unsupported quant mode")

    Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {tflite_path}")


def measure_tflite_latency(tflite_path, sample_image_np, runs=50):
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    except Exception:
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=str(tflite_path))
        except Exception:
            print("Cannot load TFLite interpreter; skipping latency measurement.")
            return None

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    # prepare sample
    inp = sample_image_np.astype(np.float32)
    if inp.ndim == 3:
        inp = np.expand_dims(inp, 0)

    total = 0.0
    for _ in range(runs):
        start = time.time()
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        total += (time.time() - start)
    avg_ms = (total / runs) * 1000.0
    print(f"TFLite avg latency over {runs} runs: {avg_ms:.2f} ms")
    return avg_ms


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    if args.dataset == "cifar10":
        print("Preparing CIFAR-10 (fast download, small dataset)...")
        ds = prepare_cifar_dataset(args.image_size, args.batch_size, split="train[:80%]")
        val_ds = prepare_cifar_dataset(args.image_size, args.batch_size, split="train[80%:90%]")
        test_ds = prepare_cifar_dataset(args.image_size, args.batch_size, split="train[90%:]")
        num_classes = 10
    elif args.dataset == "imagenet_subset":
        print("Preparing ImageNet subset via TFDS (uses small split by default)...")
        ds_raw = tfds.load("imagenet2012", split="train[:1%]", shuffle_files=True)
        def map_imagenet(ex):
            img = tf.image.resize(ex["image"], [args.image_size, args.image_size])
            img = tf.cast(img, tf.float32) / 255.0
            lbl = tf.one_hot(ex["label"], 1000)
            return img, lbl
        ds = ds_raw.map(map_imagenet).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tfds.load("imagenet2012", split="validation[:0.2%]").map(map_imagenet).batch(args.batch_size)
        num_classes = 1000
    else:
        raise NotImplementedError("local dataset flow not implemented in this script; provide coco or imagenet_subset")

    augmenter = build_augmenter(args.image_size)

    def augment(ds):
        return ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = augment(ds)

    model = build_model(num_classes, args.image_size, args.learning_rate)
    model.summary()

    callbacks = []
    ckpt_path = out_dir / "best_model.h5"
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), save_best_only=True, monitor="val_loss"))
    tb_logdir = out_dir / "logs"
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(tb_logdir)))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Save Keras SavedModel
    saved_model_dir = out_dir / "saved_model"
    model.save(str(saved_model_dir), include_optimizer=False)
    print(f"Saved SavedModel to {saved_model_dir}")

    # Convert to TFLite
    tflite_path = out_dir / "mobilenet_transfer.tflite"
    # Representative data from validation for quantization
    rep_gen = lambda: representative_data_gen(val_ds, num_samples=200)
    convert_to_tflite(str(saved_model_dir), str(tflite_path), quant=args.tflite_quant, rep_ds=rep_gen())

    # Measure latency using one sample from validation
    for batch in val_ds.take(1):
        sample_img = batch[0][0].numpy()
        break
    measure_tflite_latency(str(tflite_path), sample_img, runs=30)


if __name__ == "__main__":
    main()
"""train_mobilenet_transfer.py

Usage examples:
  python train_mobilenet_transfer.py --dataset coco --target_classes person,car,chair --epochs 5

This script:
- Loads COCO (via TensorFlow Datasets) or images from a directory
- Filters images to classes provided in `--target_classes` and builds a classification dataset
- Applies data augmentation and validation split
- Trains MobileNetV2 with transfer learning and optional fine-tuning
- Logs training metrics, checkpoints, and exports a TFLite model (with default optimizations)
- Measures TFLite inference latency
"""

import argparse
import os
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def build_augmenter(img_size):
    return tf.keras.Sequential([
        tf.keras.layers.Resizing(img_size, img_size),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom(0.08),
    ], name="data_augmentation")


def preprocess_for_model(image, label, img_size):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image * 255.0)
    return image, label


def load_coco_filtered(target_names, split="train", img_size=224, batch_size=32, val_split=0.15):
    # Load COCO with info so we can map label indices to names
    ds, info = tfds.load("coco/2017", split=split, with_info=True, shuffle_files=True)

    class_names = info.features["objects"]["label"].names
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    target_idxs = []
    for t in target_names:
        if t in name_to_idx:
            target_idxs.append(name_to_idx[t])
        else:
            print(f"Warning: target class '{t}' not found in COCO; skipping")

    if not target_idxs:
        raise ValueError("No valid target class names found in COCO dataset.")

    def keep_and_map(image, labels):
        label_idx = -1
        for i, t in enumerate(target_idxs):
            if tf.reduce_any(tf.equal(labels, t)):
                label_idx = i
                break
        return (image, label_idx)

    ds = ds.map(lambda ex: (ex["image"], ex["objects"]["label"]), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda img, labels: keep_and_map(img, labels), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda image, lbl: tf.greater_equal(lbl, 0))
    ds = ds.shuffle(50_000)
    val_count = int(val_split * 118000)  # rough; COCO train ~118k

    val_ds = ds.take(val_count)
    train_ds = ds.skip(val_count)

    return train_ds, val_ds, [t for t in target_names if t in name_to_idx]


def prepare_dataset(ds, img_size=224, batch_size=32, augment=False, cache=True, prefetch=True):
    augmenter = build_augmenter(img_size)

    def _map(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image * 255.0)
        return image, label

    if isinstance(ds, tf.data.Dataset):
        ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        if cache:
            ds = ds.cache()
        ds = ds.batch(batch_size)
        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    else:
        return ds


def build_model(num_classes, img_size=224, dropout=0.3):
    base = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base


def convert_to_tflite(model, out_path, representative_ds=None, optimize=True):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_ds is not None:
            def gen():
                for inp, _ in representative_ds.take(100):
                    yield [tf.cast(inp, tf.float32)]
            converter.representative_dataset = gen
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    Path(out_path).write_bytes(tflite_model)
    return out_path


def measure_tflite_latency(tflite_path, sample_image, runs=50, warmups=5):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    inp = sample_image.astype(np.float32)
    if inp.ndim == 3:
        inp = np.expand_dims(inp, 0)

    for _ in range(warmups):
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()

    times = []
    for _ in range(runs):
        t0 = time.time()
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times))
    p95 = float(np.percentile(times, 95))
    return mean_ms, p95


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["coco", "dir"], default="coco")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--target_classes", default="person,car,chair,sofa,bed")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--finetune_epochs", type=int, default=5)
    parser.add_argument("--model_out", default="mobilenet_transfer.h5")
    parser.add_argument("--tflite_out", default="mobilenet_transfer.tflite")
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--checkpoint", default="checkpoint.h5")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(123)

    targets = [t.strip() for t in args.target_classes.split(",") if t.strip()]

    if args.dataset == "coco":
        print("Loading COCO and filtering for target classes:", targets)
        train_raw, val_raw, available_targets = load_coco_filtered(targets, split="train", img_size=args.img_size, batch_size=args.batch_size)
        num_classes = len(available_targets)
        print(f"Found {num_classes} classes: {available_targets}")
        train_ds = prepare_dataset(train_raw, img_size=args.img_size, batch_size=args.batch_size, augment=True)
        val_ds = prepare_dataset(val_raw, img_size=args.img_size, batch_size=args.batch_size, augment=False)
    else:
        if not args.data_dir:
            raise ValueError("--data_dir is required for dataset=dir")
        print("Loading from directory:", args.data_dir)
        ds = tf.keras.preprocessing.image_dataset_from_directory(args.data_dir, image_size=(args.img_size, args.img_size), batch_size=args.batch_size, validation_split=0.15, subset='both', seed=123)
        train_ds, val_ds = ds
        class_names = train_ds.class_names
        num_classes = len(class_names)
        available_targets = class_names

    model, base = build_model(num_classes, img_size=args.img_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = []
    os.makedirs(args.logs, exist_ok=True)
    ck = tf.keras.callbacks.ModelCheckpoint(args.checkpoint, save_best_only=True, monitor='val_accuracy', mode='max')
    tb = tf.keras.callbacks.TensorBoard(log_dir=args.logs)
    csv = tf.keras.callbacks.CSVLogger(os.path.join(args.logs, 'training.csv'))
    callbacks.extend([ck, tb, csv])

    print("Starting head training...")
    model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, callbacks=callbacks)

    # Fine-tune: unfreeze last layers
    base.trainable = True
    fine_tune_at = int(len(base.layers) * 0.8)
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Starting fine-tuning...")
    model.fit(train_ds, epochs=args.epochs + args.finetune_epochs, initial_epoch=args.epochs, validation_data=val_ds, callbacks=callbacks)

    model.save(args.model_out)
    print(f"Saved Keras model to {args.model_out}")

    print("Converting to TFLite (with default optimizations)...")
    rep_ds = None
    try:
        rep_ds = val_ds.unbatch().map(lambda x, y: (x, y)).batch(1)
    except Exception:
        rep_ds = None

    tflite_path = convert_to_tflite(model, args.tflite_out, representative_ds=rep_ds, optimize=True)
    print(f"TFLite model written to {tflite_path}")

    sample_batch = next(iter(val_ds.take(1)))
    sample_img = sample_batch[0][0].numpy()
    mean_ms, p95 = measure_tflite_latency(tflite_path, sample_img)
    print(f"TFLite mean latency: {mean_ms:.2f} ms (p95: {p95:.2f} ms)")


if __name__ == '__main__':
    main()
