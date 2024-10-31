#!/bin/bash

FOLDER_ID="" # Replace it with the model parameters folder from your Google Cloud.

# 定義下載後存放的目錄
MODEL_DIR="./my_model"

# 檢查第一個模型目錄是否存在，不存在則創建
if [ ! -d "$MODEL_DIR" ]; then
    echo "Creating model directory: $MODEL_DIR"
    mkdir -p "$MODEL_DIR"
fi

# 下載Google雲端硬碟資料夾的所有內容
echo "Downloading model parameters from Google Drive..."
gdown --folder "$FOLDER_ID" -O "$MODEL_DIR"

echo "Model parameters downloaded to $MODEL_DIR"
