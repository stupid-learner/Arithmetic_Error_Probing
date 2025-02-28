from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map="auto",
)


from tqdm import tqdm
correct_answer = 0
not_number = 0

data_list = []

for i in tqdm(range(600,700)):
  for j in range(100,1000):
    prompt = str(i) + "+" + str(j) + "="
    true_answer = i+j
    prompt_tokens = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = tokenizer.decode(model.generate(**prompt_tokens, max_new_tokens=2+len(str(true_answer)))[0])
    data_list.append((i,j,outputs[5+len(prompt):].split("\n")[0]))
    try:
      if int(outputs[5+len(prompt):].split("\n")[0]) == true_answer:
        correct_answer += 1
    except:
      not_number +=0

file_name = 'data_600_to_700'
with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data_list)

print(f'Data saved to {file_name}')

print("Correct answers:" + str(correct_answer))
print("Not an integer: " + str(not_number))