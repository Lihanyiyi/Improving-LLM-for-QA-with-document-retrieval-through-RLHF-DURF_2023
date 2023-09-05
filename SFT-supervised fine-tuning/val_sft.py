import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from coati.models.opt import OPTActor
from coati.models.generation import generate
with open('val_dataset1.jsonl','r') as file:
    data = json.load(file)

tokenizer = AutoTokenizer.from_pretrained('model/output')
model = OPTActor(pretrained='facebook/opt-2.7b',
                lora_rank=8,
                checkpoint='model/output')
generate_kwargs={'repetition_penalty': 2.0, 'eos_token_id': 2, 'pad_token_id': 2}

# val_loader=DataLoader(data,batch_size=4,shuffle=True)
device = torch.device('cuda')
max_new_tokens = 100
model.to(device)
model.eval()
eval_loss = 0
print('Start')
for b in data:
    instruction = b['instruction']
    input_ids = tokenizer.encode(instruction, return_tensors='pt').to(device)
    sequences = generate(model, input_ids, max_length = len(input_ids[0])+max_new_tokens, **generate_kwargs)
    output = tokenizer.decode(sequences[0])
    b['model_outputs'] = output[len(instruction):]
    print('First done')

with open('val_dataset2.jsonl','w') as file:
    json.dump(data,file)