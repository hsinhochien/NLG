# Chinese News Summarization

## training.ipynb
### Prerequisite
1. Fix T5 FP16 Training <br>
```git clone https://github.com/huggingface/transformers.git``` <br>
```cd transformers``` <br>
```git checkout t5-fp16-no-nans``` <br>
```pip install -e .``` <br>
2. ```pip install ckiptagger rouge torch datasets tqdm scikit-learn```
### Steps to train the model
1. Data Preparation <br>
The script reads the JSON Lines data and splits it into training and validation sets.
2. Tokenization <br>
The input text and labels are tokenized using the MT5 tokenizer.
3. Model Training <br>
The MT5 model is trained using the specified training arguments, including learning rate, batch size, and number of epochs.
4. Inference <br>
The script performs inference using the trained model and experiments with different combinations of generation strategies.

## inference.py
This file is used to perform inference on ```/path/to/input.jsonl``` using the model I trained and the generation strategy I chose.

## download.sh
Run this file by executing ```bash ./download.sh```. <br>
This will download the folder ```./my_model```, which contains the model parameters for my model.

## run.sh
Run this file by executing ```bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl```. <br>
This will execute the ```inference.py``` file.