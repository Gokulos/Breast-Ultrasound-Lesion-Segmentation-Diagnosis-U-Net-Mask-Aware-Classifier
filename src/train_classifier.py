import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from .config import DATASET_ROOT, IMG_SIZE, UNET_PATH, CLF_PATH, SEED, IDX2CLASS
from .data import load_busi_dataset
from .losses import bce_dice_loss, dice_coef
from .classifier_model import build_mask_classifier

def main():
    os.makedirs(os.path.dirname(CLF_PATH), exist_ok=True)

    # Load all classes for classification
    X_all, Y_mask_all, y_all, has_mask_all = load_busi_dataset(DATASET_ROOT, IMG_SIZE)

    Xc_train, Xc_val, yc_train, yc_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
    )

    # Load trained U-Net and generate predicted masks after splitting (avoid leakage)
    unet = tf.keras.models.load_model(
        UNET_PATH,
        custom_objects={"bce_dice_loss": bce_dice_loss, "dice_coef": dice_coef}
    )

    pred_train = unet.predict(Xc_train, batch_size=8, verbose=1)
    pred_val = unet.predict(Xc_val, batch_size=8, verbose=1)

    pred_train_bin = (pred_train > 0.5).astype(np.float32)
    pred_val_bin = (pred_val > 0.5).astype(np.float32)

    X2_train = np.concatenate([Xc_train, pred_train_bin], axis=-1)  # (N,H,W,2)
    X2_val = np.concatenate([Xc_val, pred_val_bin], axis=-1)

    clf = build_mask_classifier((IMG_SIZE, IMG_SIZE, 2), num_classes=3)
    clf.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CLF_PATH, save_best_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor="val_accuracy", mode="max"),
    ]

    clf.fit(
        X2_train, yc_train,
        validation_data=(X2_val, yc_val),
        epochs=60,
        batch_size=16,
        callbacks=callbacks
    )

    print(f"Saved best classifier to: {CLF_PATH}")

    # Quick eval
    clf_best = tf.keras.models.load_model(CLF_PATH)
    probs = clf_best.predict(X2_val, verbose=0)
    preds = np.argmax(probs, axis=1)

    target_names = [IDX2CLASS[i] for i in range(3)]
    print("\nClassification report:\n")
    print(classification_report(yc_val, preds, target_names=target_names))
    print("Confusion matrix:\n", confusion_matrix(yc_val, preds))

if __name__ == "__main__":
    main()
