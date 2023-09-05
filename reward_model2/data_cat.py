import json
model_name = 'facebook/opt-350m'
max_length = 2048
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

file_path="data/val_dataset2_first140.json"
with open(file_path,"r") as fccfile:
  data1=json.load(fccfile)
  # print(json.dumps(data400,indent=4))

file_path="data/val_dataset2_last140_label.json"
with open(file_path,"r") as fccfile:
  data2=json.load(fccfile)

data = data1 + data2
dict_labels={'hallucination':1,"relevant_truth":2,"unknown":3,"related_info":4,"irrelevant":0}

c = 0
l = []
for i in data:
  str_answer=i["model_outputs"]
  input_ids = tokenizer(str_answer)['input_ids']
  d = [tokenizer.decode(i) for i in input_ids]
  #print(d)
  str_label = [0 for i in range(len(str_answer)+1)]
  str_label[-1] = 3
  lst_label=[0 for i in range(len(d))]
  lst_label[0] = 3
  if "label" in i.keys():
    labels=i["label"]
      # print(labels)
    for k in labels:
      tag_name=k['labels'][0]
      tag_code=dict_labels[tag_name]
      str_label[k['start']:k['end']] = [tag_code]*(k['end']-k['start'])
    count = 0
    for j in range(1,len(d)):
      if d[j] not in str_answer:
        lst_label[j] = 3
      else:
        #print(count,d[j],end=' _ ')
        try:
          lst_label[j] = str_label[count+1]
        except:
          lst_label[j:] = [3]*(len(lst_label)-j)
          c += 1
          l.append(i['id'])
          break
        count += len(d[j])
      #print(question,str_answer)
  #print('')
  for j in range(len(d)):
    if d[j] not in str_answer:
      lst_label[j] = 3
  i['model_labels'] = lst_label
  #print(lst_label)
  

print(len(data))

with open('data/val_dataset3.jsonl','w') as file:
  json.dump(data,file)

with open('data/val_dataset3.json','w') as file:
  json.dump(data,file)