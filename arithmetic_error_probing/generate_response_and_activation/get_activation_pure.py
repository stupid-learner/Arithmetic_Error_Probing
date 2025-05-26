import torch
import os
from general_ps_utils import ModelAndTokenizer
from utils import *
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import gc
import sys
login(token=os.environ.get('HUGGINGFACE_HUB_TOKEN'))

n_shots = int(sys.argv[1])

model_name = sys.argv[2]

result_dic = load_model_result_dic(f"{model_name.split('/')[-1]}_{n_shots}_shots_3_digit_sum_output")
prompt_with_correct_answer = []
prompt_with_wrong_answer = [] 

for i in result_dic:
    if i[0]+i[1] == result_dic[i]: 
        prompt_with_correct_answer.append(i)
    elif get_digit((i[0]+i[1]),3) != get_digit(result_dic[i],3): 
        prompt_with_wrong_answer.append(i)

prompt_by_third_digit_1 = [[] for _ in range(10)]
prompt_by_third_digit_2 = [[] for _ in range(10)]
for i in prompt_with_correct_answer:
  prompt_by_third_digit_1[get_digit((i[0]+i[1]),3)].append(i)
for i in prompt_with_wrong_answer:
  prompt_by_third_digit_2[get_digit((i[0]+i[1]),3)].append(i)


samples = []
for i in range(10):
  samples += random_select_tuples(prompt_by_third_digit_1[i], min(len(prompt_by_third_digit_1[i]), 150))
  samples += random_select_tuples(prompt_by_third_digit_2[i], min(len(prompt_by_third_digit_2[i]), 150))

print(f"Using {len(samples)} samples")
operands_list = []
for i in range(100, 1000):
   for j in range(100, 1000):
      operands_list.append((i,j))

random.seed(42)
random.shuffle(operands_list)

def create_question(i,j):
  
  return f"""Calculate the sum of the following two numbers:
  First number: {i}
  Second number: {j}"""

def create_shot(i,j, index):
  system_message = "You are a helpful assistant that calculates the sum of two numbers. Always provide your answer in the format <<x+y=z>> where x is the first number, y is the second number, and z is their sum. Do not provide any additional explanation."
  message = []
  if index == 0:
    message.append({"role": "user", "content": system_message+"\n"+create_question(i,j)})
  else:
    message.append({"role": "user", "content": create_question(i,j)})
  message.append({"role": "assistant", "content": f"<<{i}+{j}={i+j}>>"})
  return message



mt = ModelAndTokenizer(
    model_name=model_name,
    use_4bit=False,
    device='cuda'
)
#mt.model.to('cuda')
tokenizer = mt.tokenizer

first_example_printed = False
num_to_hidden = {}
for i in tqdm(samples, delay=120):

    messages = []
    shots_index = 0
    while len(messages) < 2*n_shots:
      if (int(i[0]),int(i[1])) != operands_list[shots_index]:
        messages += create_shot(operands_list[shots_index][0], operands_list[shots_index][1], len(messages))
      shots_index += 1
    system_message = "You are a helpful assistant that calculates the sum of two numbers. Always provide your answer in the format <<x+y=z>> where x is the first number, y is the second number, and z is their sum. Do not provide any additional explanation."
    if n_shots != 0: 
      messages.append({"role": "user", "content": create_question(i[0], i[1])})
    else:
      messages.append({"role": "user", "content": system_message + "\n" + create_question(i[0], i[1])})
    prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        ) + f"<<{i[0]}+{i[1]}="
    if not first_example_printed:
      print(prompt)
      first_example_printed = True
    x = tokenizer.encode(prompt, return_tensors='pt')
    x = x.to(mt.model.device)
    hidden_states = mt.model(x, output_hidden_states=True).hidden_states

    new_hidden_states = tuple(tensor[:, -1:].clone() for tensor in hidden_states)

    del hidden_states

    num_to_hidden[i] = new_hidden_states
    gc.collect()
    torch.cuda.empty_cache()

torch.save(num_to_hidden, f"generate_response_and_activation/{model_name.split('/')[-1]}_num_to_hidden_{n_shots}_shots")