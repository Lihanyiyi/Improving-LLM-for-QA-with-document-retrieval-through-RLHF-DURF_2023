import json
with open('data/val_dataset3.json','r') as file:
  r_data = json.load(file)

# Commented out IPython magic to ensure Python compatibility.
# %env HOME=/content/drive/My Drive/
# %cd ~/
#model_name = 'bert-large-uncased'
#model_name = 'bert-base-uncased'
model_name = 'facebook/opt-350m'
max_length = 2048
# past_dir = 'model/large_5epoch'
# output_dir = 'model'
import torch
# from torch.utils.data import DataLoader, Dataset
# from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
#print(tokenizer)
# config = AutoConfig.from_pretrained(model_name)
# config.num_labels=4
# config.max_position_embeddings=max_length
#print(config)
#model = BertForTokenClassification.from_pretrained(model_name, config=config,ignore_mismatched_sizes=True)
#model = BertForTokenClassification.from_pretrained(past_dir)
device = torch.device("cuda")
#model.to(device)

def fix_label(tokenizer,answer,labels):
  input_ids = tokenizer(answer)['input_ids']
  d = [tokenizer.decode(i).strip() for i in input_ids]
  #print(d)
  la = answer.split()
  if len(la) != len(labels):
    print('Error')
    return
  d.pop(0)
  q = []
  new_labels = [3]*len(d)
  left = 1
  right = 1
  for i in range(len(la)):
    #print(i)
    target = la[i]
    tmp = d.pop(0)
    while tmp not in target:
      tmp = d.pop(0)
      right += 2
      target = target[1:]
      tmp = d.pop(0)
    q.append(tmp)
    right += 1
    count = len(tmp)
    while ''.join(q) != target:
      tmp = d.pop(0)
      while tmp not in target:
        tmp = d.pop(0)
        right += 2
        target = target[:count]+target[count+1:]
        tmp = d.pop(0)
      q.append(tmp)
      count += len(tmp)
      right += 1
    new_labels[left:right] = [labels[i]]*(right-left)
    left = right
    q = []
  return new_labels
    
c = 0   
for i,entry in enumerate(r_data):
  answer = entry["model_outputs"]
  labels = entry["model_labels"]
  if len(answer.split()) != len(labels):
    c += 1
    print('Error')
    continue
  # Concatenate the retrieved documents with separator tokens
  try:
    new_labels = fix_label(tokenizer,answer,labels)
  except IndexError:
    print(i)
    c += 1
    continue
  entry['model_labels'] = new_labels


print(c)
with open('data/val_dataset3_fixed.jsonl','w') as w_file:
  json.dump(r_data,w_file)

with open('data/val_dataset3_fixed.json','w') as w_file:
  json.dump(r_data,w_file)

# c = 0
# for i in data:
#   answer = i['text'][i['len_prompt']:]
#   input_ids = tokenizer(answer)['input_ids']
#   if len(input_ids) != len(i['labels']):
#     c += 1
#     print(i['id'])

