import os
if os.getcwd() != '/scratch/yl9315/reward_model':
      os.chdir('/scratch/yl9315/reward_model')

#model_name = 'facebook/opt-350m'
#model_name = 'bert-base-uncased'
model_name = 'brad1141/gpt2-finetuned-comp2'
max_length = 2048
# past_dir = 'optmodel'
output_dir = 'model/gpt2'
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import AdamW,AutoConfig,AutoTokenizer,AutoModel,GPT2ForTokenClassification
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length=max_length
print(tokenizer)
#DATA----------------
import json
with open('whole829.jsonl','r') as file:
  r_data = json.load(file)

dataset = []
for entry in r_data:
  text = entry['text']
  len_prompt = entry['len_prompt']
  answer = text[len_prompt:]
  prompt = text[:len_prompt]
  len_prompt_t = len(tokenizer(prompt)['input_ids'])
  labels = entry["labels"]
  labels = [-100]*len_prompt_t + labels
  length = len(labels)
  labels = labels + [-100]*(max_length-length)
  encoded_input = tokenizer(text, padding="max_length", truncation=True,max_length=max_length, return_tensors='pt')
  labels = torch.tensor(labels) 
  dataset.append((encoded_input.input_ids, encoded_input.attention_mask, labels))


from sklearn.model_selection import train_test_split
train_dataset, validation_dataset = train_test_split(dataset, test_size=0.1, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader=DataLoader(validation_dataset,batch_size=4,shuffle=True)


#MODEL-------------------------
config = AutoConfig.from_pretrained(model_name)
config.num_labels=4
config.max_position_embeddings=max_length
# print(config)
#model = GPT2ForTokenClassification.from_pretrained(model_name, config=config,ignore_mismatched_sizes=True)
#model = GPT2ForTokenClassification.from_pretrained(model_name)
model = GPT2ForTokenClassification.from_pretrained('model/gpt2/15epoch',config=config)
device = torch.device("cuda")
model.to(device)
print(model)

# Define loss function
loss_fn = CrossEntropyLoss()
# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Training loop
num_epochs=100
for epoch in range(16,num_epochs+1):
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
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    total_loss+=loss.item()
    loss.backward()
    optimizer.step()
    # Calculate average loss for this epoch
  epoch_loss = total_loss / len(train_loader)
  print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
  if epoch%10 == 0:
    model.save_pretrained(output_dir+'/'+str(epoch)+'epoch')
    model.eval()
    eval_loss = 0
    for b in val_loader:
      input_ids, attention_mask, labels = b
      input_ids = input_ids.squeeze(dim=1).to(device)
      attention_mask = attention_mask.squeeze(dim=1).to(device)
      labels = labels.to(device)
      outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss
      eval_loss += loss.item()
    epoch_eval_loss = eval_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {epoch_eval_loss:.4f}")
