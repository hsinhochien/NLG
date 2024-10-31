#!/bin/bash

FOLDER_ID="https://drive.google.com/drive/folders/1V4zQi8o-n86IHa1J78uWeDSdsJneGICG?usp=sharing"

# 定義下載後存放的目錄
MODEL_DIR="./my_model_test"

# 檢查第一個模型目錄是否存在，不存在則創建
if [ ! -d "$MODEL_DIR" ]; then
    echo "Creating model directory: $MODEL_DIR"
    mkdir -p "$MODEL_DIR"
fi

# 下載Google雲端硬碟資料夾的所有內容
echo "Downloading paragraph selection model parameters from Google Drive..."
gdown --folder "$FOLDER_ID" -O "$MODEL_DIR"

echo "Model parameters downloaded to $MODEL_DIR"