import numpy as np
import tensorflow as tf

# ========= CONFIG =========
DATASET_ROOT = r"/path/to/DATASET_ROOT"  # TODO: change to your dataset path
IMG_SIZE = 256

# Model output paths
UNET_PATH = "models/unet.h5"
CLF_PATH = "models/classifier.h5"

# Class mapping
CLASS_MAP = {"normal": 0, "benign": 1, "malignant": 2}
IDX2CLASS = {v: k for k, v in CLASS_MAP.items()}

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
