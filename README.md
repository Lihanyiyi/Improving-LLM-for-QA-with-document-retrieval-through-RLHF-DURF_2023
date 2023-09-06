# Improving-LLM-for-QA-with-document-retrieval-through-RLHF-DURF_2023
DURF 2023 project with Marco and Lihan(Elaine), related to LLM

This project is based on Coati and does the following:

- data generation and cleaning
- Supervised instructions fine-tuning
- Training reward model through a sequence labeling task 
- Reinforcement learning with human feedback (token unit update)

## Description of problem
Our Task is to apply a different way of the RLHF process–changing the way to give human feedback and set up the reward model, fine-tuning a large language model to do the generative question-answering task, and finally observe whether the specific reward model and the reinforcement learning process can reduce the hallucinations in the generative answers.
We use the dataset originally from the “Faulty Directory” on NYU Shanghai Official Website (link). This dataset contains the basic information of 280 professors at NYU Shanghai, including their emails, their education level, and the courses they teach. The training and validation datasets for question-answering are created manually. There are 4 types of questions in total, which are about professors’ email addresses, professors’ highest degree, courses professors teach, and which professors teach specific courses.

## Methodology
In general, the training process includes 3 stages, which is in line with the training method introduced by OpenAI, including supervised fine-tuning, reward model training and reinforcement learning via proximal policy optimization (PPO) on this reward model.(Ouyang et al. 2022) The diagram for 3 steps fine-tuning introduced by OpenAI are as follows.(Ouyang et al. 2022) 


Figure 1: A diagram illustrating the three steps of our method: (1) supervised fine-tuning (SFT), (2)
reward model (RM) training, and (3) reinforcement learning via proximal policy optimization (PPO)
on this reward model. Blue arrows indicate that this data is used to train one of our models. In Step 2,
boxes A-D are samples from our models that get ranked by labelers.

Though the general 3 steps are the same, the methods for doing the second step are different. Instead of collecting comparison data, and generating several model outputs for ranking, we use sequence-labeling techniques to train the reward model. We use label-studio and label each token with one of 5 labels, including irrelevant, hallucination, unknown, related information and relevant truth. The definitions of 5 labels are as follows. 
Irrelevant: Have no connection with the query
Hallucination: the information has direct connection with the query but it is not true
Unknown: Has a slight connection or padding token with the query, such as “Thanks!”
Related information: Not a direct answer but true and somewhat helpful to the question
Relevant truth: True and direct information that can answer the query
Each label has a corresponding score, shown in the table below.

Irrelevant
Hallucination
Unknown
Related information
Relevant truth
-5
-2
0
3
5


The reward model is built to give the probability for each label assigned to each token and calculate the reward for each token with matrix multiplication. Lastly we sum the reward for each token to get the reward for the whole answer and do step 3.  
### 3.1 Data processing
#### 3.1.1 Documents preprocessing
The documents which we will use in the retrieval-augmentation part for LLM are from the json files in the original dataset. Because the task is only about text and the information about the professors, we first delete the abundant information like unnecessary links or pictures in the json files. The deletion is based on the index position. Furthermore, we combine the information of all the professors in the json files into one jsonl files, which make it easier for data loading and document retrieval.
#### 3.1.2 Datasets generation
The format of data in the dataset is a dictionary with the key instructions, answer, and labels. The value of the instructions key contains query and retrieved documents. The answer is the answer of the query. The labels is a list including the labels [relevant_truth, related_info, unknown, hallucination, irrelevant] about all the tokens in the generated answer. 
#### 3.1.2.1 Question generation
Because our goal is to reduce hallucinations, we created 4 kinds of factual questions with a total number of 849 based on the common information that the json files provide. The templates are as follows.
Who teaches [some course]?
What courses does [some professor] teach?
What is the highest degree of [some professor]?
Tell me [some professor]’s email address. 
#### 3.1.2.2 Top-4 documents generation
To make the length of the vector input equal, we chunk the documents into size 800, and use "all-MiniLM-L6-v2" model from Huggingface to do the embedding.
By using langchain, we create a document retrieval system which returns top-4 documents related to the query. The metrics for the relevance is based on vector similarity.
#### 3.1.2.3 Ground truth answer generation
Respectively, to make the training and validation dataset we create the ground truth answer for each question based on the information from the documents. If the information can not be directly found in the documents, we generate the answers showing that we don’t have relevant information. The answer template for answerable questions are as follows.
	[Some professors] teach [some course].
	[Some professor] teaches [some courses].
	The highest degree of [some professor] is [some degree].
	The email address of [some professor] is [some email].
#### 3.1.2.4 Sequence label generation
The labels are manually assigned by us through the label studio. Then the labels for words are transformed to token-level ones through programming. 
### 3.2 Model Creation & Evaluation
For our language model, we use facebook/opt-2.7b as our base and test its zero-shot capacity on the problem set. The result was intuitively bad.
We then fine-tune it with 8 LoRA ranks on our dataset with ground truth provided. The LoRA technique should prevent the model from forgetting too much of its learned ability.
The code for finetuning and validating the model can be found in the folder sft-supervised-finetuning.
#### 3.2.1 Reward model build-up and training
In our plan, there are several probable approaches towards building the reward model. 
One is the sequence-labeling technique. Since we need the reward model to provide scores to generate answers by detecting irrelevant generation and hallucination, we only need a classifier. In addition, as we want to improve on the PPO algorithm (which will be covered below) by providing a score to each of the tokens in the answer and give the model a more detailed update, this is a token classification task. 

For this method, we tried two different models. We start with a lightweight gpt2 token classifier, and then an OPTModel (opt-350m) with a linear classifier added on the top. We think that utilizing large language models is a good way since the transformers and the data they are pre-trained with are meant to let them understand natural language. The code for the model structure can be found in reward_model2/optfortoken.py, and the codes for training and validation are in the same folder.

Another method is to feed the data where each instruction has pairs of responses with one better than the other to the model and try to maximize the difference in scores between the pairs. Limited by the time, we didn’t use this method.
#### 3.2.2 Reinforcement Learning with Human Feedback(RLHF)
The third stage of the training is Reinforcement Learning with Human Feedback(RLHF). We experiment on our modified PPO algorithm which applies gradient descent to each of the tokens of the generated answer with different gradients. To do that, we modified the framework from ColossalAI(Bian et al. 2021) and allowed different rewards marked on different tokens in a response. We temporarily call it PPO+. We modified in ColossalAI/applications/Chat/coati the experience_maker, models/utils.py, and trainer/ppo.py.
We conduct two training sessions with the same reward model. In the first one, we train a model with lora_rank=8 from the initial opt-2.7b for causal language modeling. In the second, the model is trained from our fine-tuned opt with the same lora_rank. 
