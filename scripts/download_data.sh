#!/usr/bin/env bash
# Download and unpack the Kaggle Chest X-Ray (Pneumonia) dataset into data/chest_xray_data/
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Run: pip install kaggle" >&2
  exit 1
fi
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "Missing ~/.kaggle/kaggle.json (Kaggle -> Account -> Create New API Token)." >&2
  exit 1
fi

mkdir -p data
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data
unzip -q -o data/chest-xray-pneumonia.zip -d data/chest_xray_data
rm data/chest-xray-pneumonia.zip

echo "Done. Dataset is in data/chest_xray_data/chest_xray/"
