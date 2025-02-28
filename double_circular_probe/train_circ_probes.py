from circular_probe import train_circular_probe
import numpy
from tqdm import tqdm
from general_ps_utils import ModelAndTokenizer
from matplotlib import pyplot as plt
import torch
import csv
import random

#load data
folder = "arithmetic_data"
csv_files = []
for i in range(1,10):
  csv_files.append("data_{}00_to_{}00".format(i,i+1))
result = []

for csv_file in csv_files:
  with open(f"{folder}/{csv_file}", 'r', encoding='utf-8') as file:
      reader = csv.reader(file)
      i = 0
      for row in reader:
          result.append(tuple(row))
          i+=1

filtered_result = [i for i in result if (not "(" in i[2] and not "+" in i[2] and 3 <= len(i[2]) <= 4)]

result_dic = {}

for i in filtered_result:
    if int(i[0])+int(i[1]) < 1000 and int(i[2]) < 1000:
        result_dic[(int(i[0]),int(i[1]))] = int(i[2])

prompt_with_correct_answer = []
prompt_with_wrong_answer_1 = []
prompt_with_wrong_answer_2 = [] 

for i in result_dic:
    if i[0]+i[1] == result_dic[i]: prompt_with_correct_answer.append(i)
    elif (i[0]+i[1])//100 != result_dic[i] // 100: 
        if result_dic[i]//100 - (i[0]+i[1])//100 == 1: prompt_with_wrong_answer_1.append(i)
        elif result_dic[i]//100 - (i[0]+i[1])//100 == -1: prompt_with_wrong_answer_2.append(i)

#randomly select n prompts
def random_select_tuples(tuple_list, n, seed = None):
    if n > len(tuple_list):
        raise ValueError("n cannot be greater than the length of the tuple list")
    if seed is not None:
        random.seed(seed)
    return random.sample(tuple_list, n)

#sample_correct = random_select_tuples(prompt_with_correct_answer, 2000)
sample_wrong = random_select_tuples(prompt_with_wrong_answer_1, 1000) + random_select_tuples(prompt_with_wrong_answer_2, 1000)#+ random_select_tuples(prompt_with_correct_answer, 2000)  

'''
diff_hist = {}
for i in sample_wrong:
    diff = result_dic[i]//100 - (i[0]+i[1])//100
    if not diff in diff_hist: diff_hist[diff] = 1
    else: diff_hist[diff] += 1

print(diff_hist)
'''
first_digit_count = [0]*10
for i in prompt_with_wrong_answer_1:
    first_digit_count[result_dic[i]//100] += 1

print(first_digit_count)


first_digit_count = [0]*10
for i in prompt_with_wrong_answer_2:
    first_digit_count[result_dic[i]//100] += 1

print(first_digit_count)


first_digit_count = [0]*10
for i in sample_wrong:
    first_digit_count[result_dic[i]//100] += 1

print(first_digit_count)

exit()

def get_sum(i,j):
    return i+j

def get_data(i,j):
    return result_dic[(i,j)]


torch.cuda.empty_cache()

params = {
    'model_name': "google/gemma-2-2B", #"mistralai/Mistral-7B-v0.1",
    'use_4bit': False,
    'epochs': 10_000,
    'lr': 0.0005,
    'numbers': 2000,
    'batch_size': 1000,
    'exclude': 'random', # numbers to exclude from training set
    'exclude_count': 200,
    'positions': 1,
    'shuffle': True,
    'bases': [10],
    'start_layer': 0,
    'bias': False
}


print(f"Params:\n\n{params}")

if params['exclude'] == 'random':
    params['exclude'] = random_select_tuples(sample_wrong, params['exclude_count'])

mt = ModelAndTokenizer(
    model_name=params['model_name'],
    use_4bit=params['use_4bit'],
    device='cuda'
)

tokenizer = mt.tokenizer
num_to_hidden = dict()

# move device to cuda because for some reason it is not
mt.model.to('cuda')

print(f"device of model is {mt.model.device}")

for i in tqdm(sample_wrong, delay=120):

    text_for_embeddings = f"{i[0]}+{i[1]}="

    x = tokenizer.encode(text_for_embeddings, return_tensors='pt')
    x = x.to(mt.device)
    hidden_states = mt.model(x, output_hidden_states=True).hidden_states

    num_to_hidden[i] = hidden_states

# need to average over all layers per basis
print("Predict the true answer")
for basis in tqdm(params['bases']):
    print(f"Training cyclic probe on basis: {basis}")
    params['basis'] = basis
    layer_acc = []
    for layer in range(params['start_layer'], mt.num_layers):
        params['layers'] = [layer]
        acc, circular_probe = train_circular_probe(params, mt, num_to_hidden, get_sum, sample_wrong)
        print(f"Layer: {layer}, Accuracy: {acc}")


print("Predict the model output")
for basis in tqdm(params['bases']):
    print(f"Training cyclic probe on basis: {basis}")
    params['basis'] = basis
    layer_acc = []
    for layer in range(params['start_layer'], mt.num_layers):
        params['layers'] = [layer]
        acc, circular_probe = train_circular_probe(params, mt, num_to_hidden, get_data, sample_wrong)
        print(f"Layer: {layer}, Accuracy: {acc}")


