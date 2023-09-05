model_name = 'facebook/opt-350m'
max_length = 2048
output_dir = 'model/ld'
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import AdamW,AutoConfig,AutoTokenizer,OPTModel,AutoModel,OPTPreTrainedModel,GPT2ForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from optfortoken import OPTForTokenClassification
tokenizer = AutoTokenizer.from_pretrained(model_name)

#DATA----------------
import json
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
  encoded_input = tokenizer(text, padding="max_length", truncation=True,max_length=max_length, return_tensors='pt')
  labels = torch.tensor(labels) 
  dataset.append((encoded_input.input_ids, encoded_input.attention_mask, labels))

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
  encoded_input = tokenizer(text, padding="max_length", truncation=True,max_length=max_length, return_tensors='pt')
  labels = torch.tensor(labels) 
  dataset_ground.append((encoded_input.input_ids, encoded_input.attention_mask, labels))

dataset = dataset + dataset_ground
print('length',len(dataset))
from sklearn.model_selection import train_test_split
train_dataset, validation_dataset = train_test_split(dataset, test_size=0.1, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader=DataLoader(validation_dataset,batch_size=4,shuffle=True)


#MODEL-------------------------
# config = AutoConfig.from_pretrained(model_name)
# config.num_labels=4
# config.max_position_embeddings=max_length
# print(config)
# model = OPTModel.from_pretrained(model_name, config=config,ignore_mismatched_sizes=True)
# model = OPTModel.from_pretrained(model_name)
#model = AutoModel.from_pretrained(past_dir)
device = torch.device("cuda")
#model.to(device)

# load pre-trained model config
config = AutoConfig.from_pretrained(model_name)
config.num_labels=5
print(config)
# load the pretrained weight (up to the GPT2 output from the pretrained transformers)
model = OPTForTokenClassification.from_pretrained(model_name, config=config)
#model = OPTForTokenClassification.from_pretrained('model/opt/20epoch',config=config)
# send the model to GPU memory
model = model.to(device)

print(model)
# Define loss function
loss_fn = CrossEntropyLoss()
# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Training loop
num_epochs=100
for epoch in range(0,num_epochs+1):
  model.train()  # Set the model to training mode
  total_loss = 0
  for batch in train_loader:
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.squeeze(dim=1).to(device)
    attention_mask = attention_mask.squeeze(dim=1).to(device)
    labels = labels.to(device)
    #print("Labels shape:", labels.shape)
    #input_ids,attention_mask,labels = batch
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,return_dict=True)
    loss = outputs.loss
    total_loss+=loss.item()
    loss.backward()
    optimizer.step()
    # Calculate average loss for this epoch
  epoch_loss = total_loss / len(train_loader)
  print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")
  if epoch%10 == 0:
    model.save_pretrained(output_dir+'/'+str(epoch)+'epoch')

for i in range(0,101,10):
  model = OPTForTokenClassification.from_pretrained(output_dir+'/'+str(i)+'epoch',config=config)
  model.to(device)
  model.eval()
  eval_loss = 0
  for b in val_loader:
    input_ids, attention_mask, labels = b
    input_ids = input_ids.squeeze(dim=1).to(device)
    attention_mask = attention_mask.squeeze(dim=1).to(device)
    labels = labels.to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,return_dict=True)
    loss = outputs.loss
    eval_loss += loss.item()
  epoch_eval_loss = eval_loss / len(val_loader)
  print(f"Epoch {epoch}/{num_epochs}, Val Loss: {epoch_eval_loss:.4f}")
