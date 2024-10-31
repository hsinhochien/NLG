#!/bin/bash

# 檢查是否有足夠的參數輸入
if [ "$#" -ne 2 ]; then
    echo "Usage: bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl"
    exit 1
fi

# 設定變數
INPUT_PATH="${1}"
OUTPUT_PATH="${2}"

# 執行 inference.py 並傳遞參數
echo "Running inference with input: $INPUT_PATH, output: $OUTPUT_PATH"
python3 inference.py --input "$INPUT_PATH" --output "$OUTPUT_PATH"
