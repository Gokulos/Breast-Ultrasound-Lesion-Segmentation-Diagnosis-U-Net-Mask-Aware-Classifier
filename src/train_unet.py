import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .config import DATASET_ROOT, IMG_SIZE, UNET_PATH, SEED
from .data import load_busi_dataset
from .unet_model import build_unet
from .losses import bce_dice_loss, dice_coef

def main():
    os.makedirs(os.path.dirname(UNET_PATH), exist_ok=True)

    X_all, Y_mask_all, y_all, has_mask_all = load_busi_dataset(DATASET_ROOT, IMG_SIZE)
    print("Loaded:", X_all.shape, Y_mask_all.shape, "masks:", int(has_mask_all.sum()))

    # Train segmentation only on samples that have real masks
    X_seg = X_all[has_mask_all]
    Y_seg = Y_mask_all[has_mask_all]

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_seg, Y_seg, test_size=0.2, random_state=SEED
    )

    unet = build_unet((IMG_SIZE, IMG_SIZE, 1))
    unet.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=bce_dice_loss,
        metrics=[dice_coef],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(UNET_PATH, save_best_only=True, monitor="val_dice_coef", mode="max"),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_dice_coef", mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor="val_dice_coef", mode="max"),
    ]

    unet.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=60,
        batch_size=8,
        callbacks=callbacks
    )

    print(f"Saved best U-Net to: {UNET_PATH}")

if __name__ == "__main__":
    main()
