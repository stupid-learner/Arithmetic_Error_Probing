#python -m arithmetic_error_probing.train.train_probes_on_pure_wo_equation google/gemma-2-2b-it gemma-2-2b-it_2_shots_3_digit_sum_wo_equation_output arithmetic_error_probing/generate_response_and_activation/gemma-2-2b-it_sum_start_num_to_hidden_2_shots_wo_equation gemma-2-2b-it_sum_wo_equation_probing_results_2_shots sum


import torch
import os
from arithmetic_error_probing.generate_response_and_activation.general_ps_utils import ModelAndTokenizer
from arithmetic_error_probing.utils import *
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import gc
import sys
login(token=os.environ.get('HUGGINGFACE_HUB_TOKEN'))

n_shots = int(sys.argv[1])

model_name = sys.argv[2]

#arithmetic_mode = "difference"
arithmetic_mode = sys.argv[3]

digit_index = sys.argv[4]

if arithmetic_mode == "sum":
    arithmetic_operator = "+"
    arithmetic_function = lambda a,b: a+b
elif arithmetic_mode == "difference":
    arithmetic_operator = "-"
    arithmetic_function = lambda a,b: a-b
elif arithmetic_mode == "product":
    arithmetic_operator = "*"
    arithmetic_function = lambda a,b: a*b

result_dic = load_model_result_dic(f"{model_name.split('/')[-1]}_{n_shots}_shots_3_digit_{arithmetic_mode}_wo_equation_output")
prompt_with_correct_answer = []
prompt_with_wrong_answer = [] 

for i in result_dic:
    if arithmetic_function(i[0],i[1]) == result_dic[i]: 
        prompt_with_correct_answer.append(i)
    elif get_digit(arithmetic_function(i[0],i[1]),digit_index) != get_digit(result_dic[i],digit_index): 
        prompt_with_wrong_answer.append(i)

prompt_by_third_digit_1 = [[] for _ in range(10)]
prompt_by_third_digit_2 = [[] for _ in range(10)]
for i in prompt_with_correct_answer:
  prompt_by_third_digit_1[get_digit(arithmetic_function(i[0],i[1]),digit_index)].append(i)
for i in prompt_with_wrong_answer:
  prompt_by_third_digit_2[get_digit(arithmetic_function(i[0],i[1]),digit_index)].append(i)


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
  
  return f"""Calculate the {arithmetic_mode} of the following two numbers:
  First number: {i}
  Second number: {j}"""

def create_shot(i, j, index):
    system_message = f"You are a helpful assistant that calculates the {arithmetic_mode} of two numbers. Always respond with 'the answer is [number]' where [number] is the correct result. Do not provide any additional explanation."
    message = []
    if index == 0:
        message.append({"role": "user", "content": system_message + "\n" + create_question(i, j)})
    else:
        message.append({"role": "user", "content": create_question(i, j)})
    message.append({"role": "assistant", "content": f"the answer is {arithmetic_function(i,j)}"})
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
    system_message = f"You are a helpful assistant that calculates the {arithmetic_mode} of two numbers. Always respond with 'the answer is [number]' where [number] is the correct result. Do not provide any additional explanation."
    if n_shots != 0: 
      messages.append({"role": "user", "content": create_question(i[0], i[1])})
    else:
      messages.append({"role": "user", "content": system_message + "\n" + create_question(i[0], i[1])})
    prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        ) + "the answer is "
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

torch.save(num_to_hidden, f"arithmetic_error_probing/generate_response_and_activation/{model_name.split('/')[-1]}_{arithmetic_mode}_{digit_index}_num_to_hidden_{n_shots}_shots_wo_equation")