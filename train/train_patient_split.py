import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

from dataset_builder import build_dataset, CLASS_NAMES


def build_model(n_steps: int, n_features: int, n_classes: int):
    inputs = layers.Input(shape=(n_steps, n_features))

    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)

    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--artifacts-dir', default='../artifacts')
    parser.add_argument('--n-steps', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups = build_dataset(args.data_dir, n_steps=args.n_steps)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

    model = build_model(X_train.shape[1], X_train.shape[2], len(CLASS_NAMES))

    early = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-5)

    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=256,
        callbacks=[early, reduce],
        class_weight=class_weight_dict,
        verbose=1,
    )

    probs = model.predict(X_test_scaled, verbose=0)
    preds = probs.argmax(axis=1)

    report = classification_report(
        y_test,
        preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        'accuracy': float((preds == y_test).mean()),
        'macro_f1': float(f1_score(y_test, preds, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, preds)),
        'confusion_matrix': confusion_matrix(y_test, preds).tolist(),
        'class_distribution_train': dict(Counter(map(int, y_train))),
        'class_distribution_test': dict(Counter(map(int, y_test))),
        'n_train_records': int(len(set(groups_train.tolist()))),
        'n_test_records': int(len(set(groups_test.tolist()))),
        'model_version': 'patient-split-1.0.0',
        'input_type': '15_rr_intervals',
        'feature_channels': ['rr', 'rr_diff', 'rolling_mean', 'rr_zscore'],
        'split_strategy': 'GroupShuffleSplit_by_record',
        'report': report,
    }

    model.save(artifacts_dir / 'model.keras')
    joblib.dump(scaler, artifacts_dir / 'scaler.joblib')
    (artifacts_dir / 'metadata.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    print(json.dumps({k: v for k, v in metrics.items() if k not in {'report', 'confusion_matrix'}}, indent=2))


if __name__ == '__main__':
    main()
