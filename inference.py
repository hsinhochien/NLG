import json
import pandas as pd
from transformers import MT5ForConditionalGeneration, AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from datasets import Dataset
import torch

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)  # 將每一行轉換為JSON對象
            data.append(json_data)
    return data

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for text summarization model")
    parser.add_argument('--input', required=True, help='Path to input.jsonl')
    parser.add_argument('--output', required=True, help='Path to output.jsonl')
    return parser.parse_args()

args = parse_args()
data = read_jsonl(args.input)

checkpoint = "./my_model"
model = MT5ForConditionalGeneration.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

if torch.cuda.is_available():
    print("GPU is available. Running on:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("GPU is not available. Running on CPU.")
    device = torch.device("cpu")

# 將模型移動到 GPU
model = model.to(device)

def preprocess_function(examples):
    model_inputs = tokenizer(examples["maintext"], max_length=256, truncation=True, padding=True)
    return model_inputs

test_dataset = Dataset.from_pandas(pd.DataFrame(data))
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# 創建 DataLoader
collator = DataCollatorForSeq2Seq(tokenizer, model=model)
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=16, collate_fn=collator)

def generate_predictions(batch):
    inputs = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=inputs, 
        attention_mask=attention_mask, 
        max_length=64, 
        num_beams=7
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# 生成預測並保存到 .jsonl 文件中
predictions = []

with open(args.output, "w", encoding="utf-8") as outfile:
    for i, batch in enumerate(tqdm(test_dataloader, desc="Generating predictions")):
        preds = generate_predictions(batch)
        
        # 獲取對應的 id
        batch_ids = test_dataset['id'][i * test_dataloader.batch_size: (i + 1) * test_dataloader.batch_size]

        # 寫入結果
        for j, pred in enumerate(preds):
            output = {
                "title": pred,
                "id": batch_ids[j]
            }
            outfile.write(json.dumps(output, ensure_ascii=False) + "\n")

print(f"Results saved to {args.output}")
