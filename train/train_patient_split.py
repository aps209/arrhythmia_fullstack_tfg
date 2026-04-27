from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models

from dataset_builder import CLASS_NAMES, build_dataset


ARCHITECTURE_SPECS = {
    "conv1d_bilstm": {
        "label": "Conv1D + BiLSTM",
        "architecture_name": "Conv1D_BiLSTM_ECG",
        "subdir": "conv1d_bilstm",
        "model_version": "ecg-conv1d-bilstm-1.0.0",
    },
    "lstm": {
        "label": "LSTM",
        "architecture_name": "LSTM_ECG",
        "subdir": "lstm",
        "model_version": "ecg-lstm-1.0.0",
    },
    "simple_rnn": {
        "label": "Simple RNN",
        "architecture_name": "SimpleRNN_ECG",
        "subdir": "simple_rnn",
        "model_version": "ecg-simple-rnn-1.0.0",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_conv1d_bilstm_model(window_size: int, n_features: int, n_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(window_size, n_features), name="ecg_segment")

    x = layers.Conv1D(32, kernel_size=7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    residual = x
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.Activation("relu")(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="aami_softmax")(x)
    return models.Model(inputs, outputs, name="Conv1D_BiLSTM_ECG")


def build_lstm_model(window_size: int, n_features: int, n_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(window_size, n_features), name="ecg_segment")
    x = layers.LSTM(96, return_sequences=True)(inputs)
    x = layers.Dropout(0.25)(x)
    x = layers.LSTM(48)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="aami_softmax")(x)
    return models.Model(inputs, outputs, name="LSTM_ECG")


def build_simple_rnn_model(window_size: int, n_features: int, n_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(window_size, n_features), name="ecg_segment")
    x = layers.SimpleRNN(96, return_sequences=True)(inputs)
    x = layers.Dropout(0.25)(x)
    x = layers.SimpleRNN(48)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="aami_softmax")(x)
    return models.Model(inputs, outputs, name="SimpleRNN_ECG")


def build_model(architecture: str, window_size: int, n_features: int, n_classes: int) -> tf.keras.Model:
    if architecture == "conv1d_bilstm":
        model = build_conv1d_bilstm_model(window_size, n_features, n_classes)
    elif architecture == "lstm":
        model = build_lstm_model(window_size, n_features, n_classes)
    elif architecture == "simple_rnn":
        model = build_simple_rnn_model(window_size, n_features, n_classes)
    else:
        raise ValueError(f"Arquitectura no soportada: {architecture}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def split_groups(X: np.ndarray, y: np.ndarray, groups: np.ndarray, test_size: float, seed: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return train_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--artifacts-dir", default="../artifacts")
    parser.add_argument("--architecture", default="conv1d_bilstm", choices=sorted(ARCHITECTURE_SPECS.keys()))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-fs", type=int, default=250)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--pre-samples", type=int, default=96)
    parser.add_argument("--lead-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    spec = ARCHITECTURE_SPECS[args.architecture]
    artifacts_root = Path(args.artifacts_dir)
    artifacts_dir = artifacts_root / spec["subdir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups = build_dataset(
        data_dir=args.data_dir,
        target_fs=args.target_fs,
        window_size=args.window_size,
        pre_samples=args.pre_samples,
        preferred_lead=args.lead_name,
    )
    if len(X) == 0:
        raise RuntimeError("No se generó ningún segmento. Revisa el dataset WFDB y las anotaciones `.atr`.")

    train_val_idx, test_idx = split_groups(X, y, groups, test_size=0.2, seed=args.seed)
    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = y[train_val_idx], y[test_idx]
    groups_train_val, groups_test = groups[train_val_idx], groups[test_idx]

    train_idx, val_idx = split_groups(X_train_val, y_train_val, groups_train_val, test_size=0.15, seed=args.seed + 1)
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
    groups_train, groups_val = groups_train_val[train_idx], groups_train_val[val_idx]

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}

    model = build_model(args.architecture, X.shape[1], X.shape[2], len(CLASS_NAMES))

    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(artifacts_dir / "model.keras"),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )
    early = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    reduce = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[checkpoint, early, reduce],
        class_weight=class_weight_dict,
        verbose=1,
    )

    best_model = tf.keras.models.load_model(artifacts_dir / "model.keras")
    probs = best_model.predict(X_test, verbose=0)
    preds = probs.argmax(axis=1)

    report = classification_report(
        y_test,
        preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class_sensitivity = recall_score(y_test, preds, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=list(range(len(CLASS_NAMES))))
    specificities = []
    for class_idx in range(len(CLASS_NAMES)):
        tp = cm[class_idx, class_idx]
        fn = cm[class_idx, :].sum() - tp
        fp = cm[:, class_idx].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)

    metrics = {
        "model_key": args.architecture,
        "model_label": spec["label"],
        "model_version": spec["model_version"],
        "architecture_name": spec["architecture_name"],
        "input_type": "wfdb_ecg_signal",
        "target_fs": args.target_fs,
        "window_size": args.window_size,
        "pre_samples": args.pre_samples,
        "class_names": CLASS_NAMES,
        "lead_preference": args.lead_name or "auto",
        "split_strategy": "GroupShuffleSplit_by_record",
        "accuracy": float((preds == y_test).mean()),
        "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "confusion_matrix": cm.tolist(),
        "class_distribution_train": dict(Counter(map(int, y_train))),
        "class_distribution_val": dict(Counter(map(int, y_val))),
        "class_distribution_test": dict(Counter(map(int, y_test))),
        "n_train_records": int(len(set(groups_train.tolist()))),
        "n_val_records": int(len(set(groups_val.tolist()))),
        "n_test_records": int(len(set(groups_test.tolist()))),
        "sensitivity_per_class": {CLASS_NAMES[i]: float(per_class_sensitivity[i]) for i in range(len(CLASS_NAMES))},
        "specificity_per_class": {CLASS_NAMES[i]: float(specificities[i]) for i in range(len(CLASS_NAMES))},
        "report": report,
        "history": {
            "loss": [float(v) for v in history.history.get("loss", [])],
            "val_loss": [float(v) for v in history.history.get("val_loss", [])],
            "accuracy": [float(v) for v in history.history.get("accuracy", [])],
            "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
        },
        "seed": args.seed,
    }

    (artifacts_dir / "metadata.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"history", "report", "confusion_matrix"}}, indent=2))


if __name__ == "__main__":
    main()
