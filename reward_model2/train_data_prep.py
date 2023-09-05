model_name = 'facebook/opt-350m'
max_length = 2048
import torch
from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained(model_name)
with open('data/val_dataset3.jsonl','r') as file:
  r_data = json.load(file)

dataset = []
for entry in r_data:
  prompt = entry['instruction']
  text = prompt+entry['model_outputs']
  len_prompt_t = len(tokenizer(prompt)['input_ids'])
  labels = [-100]*len_prompt_t + entry["model_labels"]
  length = len(labels)
  labels = labels + [-100]*(max_length-length)
  encoded_input = tokenizer(text, padding="max_length", truncation=True,max_length=max_length)
  dataset.append((encoded_input.input_ids, encoded_input.attention_mask, labels))
  
with open('data/train_s.jsonl','w') as file:
  json.dump(dataset,file)

with open('data/ppo_dataset1.jsonl','r') as file:
  p_data = json.load(file)

with open('data/sft_dataset1.jsonl','r') as file:
  s_data = json.load(file)

ground_data = r_data+p_data+s_data
dataset_ground = []
for entry in ground_data:
  prompt = entry['instruction']
  text = prompt+entry['output']
  len_prompt_t = len(tokenizer(prompt)['input_ids'])
  labels = [-100]*len_prompt_t + entry["labels"]
  length = len(labels)
  labels = labels + [-100]*(max_length-length)
  encoded_input = tokenizer(text, padding="max_length", truncation=True,max_length=max_length)
  dataset_ground.append((encoded_input.input_ids, encoded_input.attention_mask, labels))

dataset = dataset + dataset_ground

with open('data/train_l.jsonl','w') as file:
  json.dump(dataset,file)