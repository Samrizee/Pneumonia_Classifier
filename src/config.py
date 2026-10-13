"""Central configuration for the chest X-ray CNN project."""

import os

# Project root = parent of the src/ folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where the Kaggle dataset lives (see data/README.md). Override with the
# CHEST_XRAY_DATA environment variable if you keep it somewhere else.
DATA_ROOT = os.environ.get(
    "CHEST_XRAY_DATA",
    os.path.join(PROJECT_ROOT, "data", "chest_xray_data", "chest_xray"),
)

# Where figures / metrics / the trained model are written.
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # label 0 and label 1

IMG_SIZE = 150          # the exercise suggests 150x150 as a reasonable start
BATCH_SIZE = 32
VAL_FRACTION = 0.15     # 85 / 15 train / validation split
MAX_EPOCHS = 25         # exercise: 20-30, early stopping decides the real cutoff
PATIENCE = 4            # exercise: 3-5
SEED = 42
