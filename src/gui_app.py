# Tkinter GUI to run end-to-end inference: image -> mask + class
# Run: python -m src.gui_app

import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt

from .infer import load_models, predict_mask_and_class, preprocess_single, overlay_mask, LABELS

def run_gui():
    unet, clf = load_models()

    def load_image_dialog():
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("PNG Images", "*.png")]
        )
        if not file_path:
            return

        pred_mask, cls_id, probs, cls_name = predict_mask_and_class(file_path, unet, clf)

        img = preprocess_single(file_path)[0, ..., 0]
        overlay = overlay_mask(img, pred_mask)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img, cmap="gray"); ax[0].axis("off"); ax[0].set_title("Input")
        ax[1].imshow(pred_mask > 0.5, cmap="gray"); ax[1].axis("off"); ax[1].set_title("Pred Mask (thr)")
        ax[2].imshow(overlay[..., ::-1]); ax[2].axis("off")
        ax[2].set_title(f"Pred class: {cls_name}\nprobs={np.round(probs, 3)}")
        plt.show(block=False)

    root = tk.Tk()
    root.title("Breast Ultrasound: Mask + Class")
    root.geometry("420x220")

    tk.Label(root, text="Select an Ultrasound Image (.png)", font=("Arial", 12)).pack(pady=12)
    tk.Button(root, text="Choose Image", command=load_image_dialog, font=("Arial", 12)).pack(pady=10)
    tk.Label(root, text="Outputs: predicted lesion mask + class probabilities", font=("Arial", 10)).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    run_gui()
