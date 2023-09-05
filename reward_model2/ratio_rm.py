import torch
import torch.nn as nn
from .optfortoken import OPTForTokenClassification
from transformers import AutoConfig, AutoTokenizer, GPT2ForTokenClassification

dict_labels={'hallucination':1,"relevant_truth":2,"unknown":3,"related_info":4,"irrelevant":0}

class RatioRM(nn.Module):
    def __init__(self,model,score_matrix=torch.Tensor([-5,-2,5,0,3])):
        super().__init__()
        self.model = model
        self.device = torch.cuda.current_device()
        #self.model.to(torch.float16).to(self.device)
        self.softmax = nn.Softmax(dim=-1)
        #self.softmax.to(self.device)
        # 0:irrelevant, 1:hallucination, 2:relevant_truth, 3:unknown
        self.score_matrix = score_matrix.type(torch.float16).to(self.device)
        #self.score_matrix.to(torch.float16).to(self.device)
    
    def compute_reward(self,prob):
        return torch.matmul(prob,self.score_matrix)

    def forward(self,input_ids = None, attention_mask = None,action_mask = None, num_actions = None, labels = None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits
        prob = self.softmax(logits)
        #prob.to(torch.float16).to(torch.cuda.current_device())
        #print(prob)
        #print(self.score_matrix)
        score = self.compute_reward(prob)
        score = score[:,-num_actions:]
        score = score*action_mask
        return score


