model_name = 'facebook/opt-350m'
max_length = 2048
output_dir = 'model/sd'
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import AdamW,AutoConfig,AutoTokenizer,OPTModel,OPTPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from optfortoken import OPTForTokenClassification
tokenizer = AutoTokenizer.from_pretrained(model_name)

#DATA----------------
import json
#with open('data/train_l.jsonl','r') as file:
with open('data/train_s.jsonl','r') as file:
  dataset = json.load(file)

for i in range(len(dataset)):
    a,b,c = dataset[i]
    dataset[i] = (torch.tensor(a),torch.tensor(b),torch.tensor(c))

print('length',len(dataset))
from sklearn.model_selection import train_test_split
train_dataset, validation_dataset = train_test_split(dataset, test_size=0.1, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader=DataLoader(validation_dataset,batch_size=4,shuffle=True)

device = torch.device("cuda")
#model.to(device)

# load pre-trained model config
config = AutoConfig.from_pretrained(model_name)
config.num_labels=5
print(config)
# load the pretrained weight (up to the GPT2 output from the pretrained transformers)
model = OPTForTokenClassification.from_pretrained('model/50epoch', config=config)
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
for epoch in range(51,num_epochs+1):
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

for i in range(60,101,10):
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
