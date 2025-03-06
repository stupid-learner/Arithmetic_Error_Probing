import torch
import os
from model import CircularErrorDetector
from general_ps_utils import ModelAndTokenizer
from utlis import *
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

torch.manual_seed(42)

def train_error_detector(num_to_hidden, layer_index, model_result, training_prompt, testing_prompt, epochs = 10_000, batch_size = 1000, lr = 0.005):
    device = "cuda"
    embedding_size = list(num_to_hidden.values())[0][0].shape[-1]
    circular_error_detector = CircularErrorDetector(embedding_size, 10, False)
    circular_error_detector = circular_error_detector.to(device)

    training_x = []
    training_y = []
    for i,j in training_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        training_x.append(x)
        if (i+j)//100 == model_result[(i,j)]//100:
            training_y.append(torch.tensor(1))
        else:
            training_y.append(torch.tensor(0))
    training_x = torch.stack(training_x)
    training_y = torch.stack(training_y)

    perm = torch.randperm(training_x.shape[0])
    training_x = training_x[perm]
    training_y = training_y[perm]
    training_x = training_x.to("cuda")
    training_y = training_y.to("cuda").float()


    testing_x = []
    testing_y = []
    for i,j in testing_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        testing_x.append(x)
        if (i+j)//100 == model_result[(i,j)]//100:
            testing_y.append(torch.tensor(1))
        else:
            testing_y.append(torch.tensor(0))
    testing_x = torch.stack(testing_x)
    testing_y = torch.stack(testing_y)
    testing_x = testing_x.to("cuda")
    testing_y = testing_y.to("cuda").float()

    assert training_x.shape[0] == training_y.shape[0], f"{training_x.shape=}, {training_y.shape=}"
    assert testing_x.shape[0] == testing_y.shape[0], f"{testing_x.shape=}, {testing_y.shape=}"

    optimizer = torch.optim.Adam(circular_error_detector.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    for epoch in range(epochs):
        
        for i in range(0, len(training_x), batch_size):
            X_batch = training_x[i:i+batch_size]
            Y_batch = training_y[i:i+batch_size]

            optimizer.zero_grad()
            Y_pred = circular_error_detector(X_batch)

            assert Y_pred.shape == Y_batch.shape, f"{Y_pred.shape=}, {Y_batch.shape=}"
            assert Y_pred.device == Y_batch.device, f"{Y_pred.device=}, {Y_batch.device=}"

            loss = loss_fn(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()

    predicted_result = circular_error_detector(testing_x)
    predicted_labels = (predicted_result > 0.5).int()
    accuracy = (predicted_labels == testing_y).float().mean()

    print(f"The accuracy on layer {layer_index} is {accuracy}.")

    return circular_error_detector, predicted_result


result_dic = load_model_result_dic()

prompt_with_correct_answer = []
prompt_with_wrong_answer_1 = [] #model_result//100 = correct_answer//100 + 1
prompt_with_wrong_answer_2 = [] #model_result//100 = correct_answer//100 - 1

#error_count = 0
for i in result_dic:
    #error_count += 1
    if i[0]+i[1] == result_dic[i]: 
        #error_count -= 1
        prompt_with_correct_answer.append(i)
    elif (i[0]+i[1])//100 != result_dic[i] // 100: 
        if result_dic[i]//100 - (i[0]+i[1])//100 == 1: prompt_with_wrong_answer_1.append(i)
        elif result_dic[i]//100 - (i[0]+i[1])//100 == -1: prompt_with_wrong_answer_2.append(i)
#print(f"number of errors is {error_count}")

samples = []
samples += get_balanced_data(prompt_with_correct_answer, 175)
samples += get_balanced_data(prompt_with_wrong_answer_1)
samples += get_balanced_data(prompt_with_wrong_answer_2)

print(f"Use {len(samples)} for training.")

samples_train = random_select_tuples(samples, 2000)
samples_test = list(set(samples) - set(samples_train))



#load the model
mt = ModelAndTokenizer(
    model_name="google/gemma-2-2B",
    use_4bit=False,
    device='cuda'
)
mt.model.to('cuda')
tokenizer = mt.tokenizer

if not os.path.exists("double_circular_probe/num_to_hidden"):
    num_to_hidden = {}
    for i in tqdm(samples, delay=120):

        text_for_embeddings = f"{i[0]}+{i[1]}="

        x = tokenizer.encode(text_for_embeddings, return_tensors='pt')
        x = x.to(mt.model.device)
        hidden_states = mt.model(x, output_hidden_states=True).hidden_states

        num_to_hidden[i] = hidden_states
    torch.save(num_to_hidden, "double_circular_probe/num_to_hidden")
else:
    num_to_hidden = torch.load("double_circular_probe/num_to_hidden")

num_layers = mt.model.config.num_hidden_layers

if len(num_to_hidden[samples[0]]) == num_layers + 1:
    start_layer = 1
else:
    start_layer = 0

for layer_index in tqdm(range(start_layer, len(num_to_hidden[samples[0]]))):
    circular_error_detector, predicted_result = train_error_detector(num_to_hidden, layer_index, result_dic, samples_train, samples_test)
    torch.save((circular_error_detector, predicted_result), f"double_circular_probe/training_result/layer_{layer_index}")