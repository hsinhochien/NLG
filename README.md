# Chinese News Summarization
Install tw_rouge <br>
```git clone https://github.com/deankuo/ADL24-HW2.git``` <br>
```cd ADL24-HW2``` <br>
```pip install -e tw_rouge``` <br>

## training.ipynb
### Prerequisite
1. Fix T5 FP16 Training <br>
```git clone https://github.com/huggingface/transformers.git``` <br>
```cd transformers``` <br>
```git checkout t5-fp16-no-nans``` <br>
```pip install -e .``` <br>
2. ```pip install transformers torch datasets tqdm scikit-learn```
### Steps to train the model
1. Data Preparation <br>
Ensure you have the files ```train.jsonl```, ```public.jsonl``` ready.
2. Tokenization <br>
The input text and labels are tokenized using the MT5 tokenizer.
3. Model Training <br>
The MT5 model is trained using the specified training arguments, including learning rate, batch size, and number of epochs.




## inference.py
This file is used to perform inference on ```/path/to/input.jsonl``` using the model I trained and the generation strategy I chose.

## download.sh
Run this file by executing ```bash ./download.sh```. <br>
This will download the folder ```./my_model```, which contains the model parameters for my model.

## run.sh
Run this file by executing ```bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl```. <br>
This will execute the ```inference.py``` file.
