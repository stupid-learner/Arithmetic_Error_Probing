import torch
import os
from model import RidgeRegression, MultiClassLogisticRegression
from general_ps_utils import ModelAndTokenizer
from utlis import *
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login


def train_linear_probe(num_to_hidden, layer_index, x_to_y, training_prompt, testing_prompt, lambda_, epochs = 1000, lr = 0.1):
    device = "cuda"
    embedding_size = list(num_to_hidden.values())[0][0].shape[-1]
    linear_probe = RidgeRegression(embedding_size, lambda_)
    linear_probe = linear_probe.to(device)

    training_x = []
    training_y = []
    for i,j in training_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        training_x.append(x)
        training_y.append(torch.tensor(x_to_y(i,j)))
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
        testing_y.append(torch.tensor(x_to_y(i,j)))
    testing_x = torch.stack(testing_x)
    testing_y = torch.stack(testing_y)
    testing_x = testing_x.to("cuda")
    testing_y = testing_y.to("cuda").float()

    assert training_x.shape[0] == training_y.shape[0], f"{training_x.shape=}, {training_y.shape=}"
    assert testing_x.shape[0] == testing_y.shape[0], f"{testing_x.shape=}, {testing_y.shape=}"

    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = linear_probe.loss(training_x, training_y)
        loss.backward()
        optimizer.step()

    linear_probe.eval()
    with torch.no_grad():
        y_pred_train = (linear_probe(training_x)//100).long()
        y_pred_test = (linear_probe(testing_x)//100).long()
    
    y_train_class = (training_y//100).long()
    y_test_class = (testing_y//100).long()

    training_correct_predictions = (y_pred_train == y_train_class).float()
    training_accuracy = training_correct_predictions.mean().item()
    
    testing_correct_predictions = (y_pred_test == y_test_class).float()
    testing_accuracy = testing_correct_predictions.mean().item()
    
    #print(f"{training_accuracy=}")
    print(f"{testing_accuracy=}")
    

def train_error_detector(num_to_hidden, layer_index, x_to_y, training_prompt, testing_prompt, lambda_, epochs = 1000, lr = 0.1):
    #We still use Ridge regression. But we predict the difference between GT results and model outputs.
    device = "cuda"
    embedding_size = list(num_to_hidden.values())[0][0].shape[-1]
    linear_probe = RidgeRegression(embedding_size, lambda_)
    linear_probe = linear_probe.to(device)

    training_x = []
    training_y = []
    for i,j in training_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        training_x.append(x)
        training_y.append(torch.tensor(x_to_y(i,j)//100-(i+j)//100))
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
        testing_y.append(torch.tensor(x_to_y(i,j)//100-(i+j)//100))
    testing_x = torch.stack(testing_x)
    testing_y = torch.stack(testing_y)
    testing_x = testing_x.to("cuda")
    testing_y = testing_y.to("cuda").float()

    assert training_x.shape[0] == training_y.shape[0], f"{training_x.shape=}, {training_y.shape=}"
    assert testing_x.shape[0] == testing_y.shape[0], f"{testing_x.shape=}, {testing_y.shape=}"

    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = linear_probe.loss(training_x, training_y)
        loss.backward()
        optimizer.step()

    linear_probe.eval()
    with torch.no_grad():
        y_pred_train = (linear_probe(training_x)).long()
        y_pred_test = (linear_probe(testing_x)).long()
    
    y_train_class = (training_y).long()
    y_test_class = (testing_y).long()

    training_correct_predictions = (y_pred_train == y_train_class).float()
    training_accuracy = training_correct_predictions.mean().item()
    
    testing_correct_predictions = (y_pred_test == y_test_class).float()
    testing_accuracy = testing_correct_predictions.mean().item()
    print(f"{training_accuracy=}")
    print(f"{testing_accuracy=}")

def train_logistic_probe(num_to_hidden, layer_index, x_to_y, training_prompt, testing_prompt, epochs = 1000, lr = 0.1):
    device = "cuda"
    embedding_size = list(num_to_hidden.values())[0][0].shape[-1]
    logistic_probe = MultiClassLogisticRegression(embedding_size, 8)
    logistic_probe = logistic_probe.to(device)

    training_x = []
    training_y = []
    for i,j in training_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        training_x.append(x)
        training_y.append(torch.tensor(x_to_y(i,j)//100-2))
    training_x = torch.stack(training_x)
    training_y = torch.stack(training_y)

    perm = torch.randperm(training_x.shape[0])
    training_x = training_x[perm]
    training_y = training_y[perm]
    training_x = training_x.to("cuda")
    training_y = training_y.to("cuda").long()


    testing_x = []
    testing_y = []
    for i,j in testing_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        testing_x.append(x)
        testing_y.append(torch.tensor(x_to_y(i,j)//100-2))
    testing_x = torch.stack(testing_x)
    testing_y = torch.stack(testing_y)
    testing_x = testing_x.to("cuda")
    testing_y = testing_y.to("cuda").long()

    optimizer = torch.optim.Adam(logistic_probe.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = logistic_probe(training_x)
        loss = criterion(outputs, training_y)
        loss.backward()
        optimizer.step()

    logistic_probe.eval()
    with torch.no_grad():
        _, y_pred_train = torch.max((logistic_probe(training_x)),1)
        _, y_pred_test = torch.max((logistic_probe(testing_x)),1)
    
    y_train_class = (training_y).long()
    y_test_class = (testing_y).long()

    training_correct_predictions = (y_pred_train == y_train_class).float()
    training_accuracy = training_correct_predictions.mean().item()
    
    testing_correct_predictions = (y_pred_test == y_test_class).float()
    testing_accuracy = testing_correct_predictions.mean().item()
    print(f"{training_accuracy=}")
    print(f"{testing_accuracy=}")

def train_logistic_error_detector(num_to_hidden, layer_index, x_to_y, training_prompt, testing_prompt, epochs = 1000, lr = 0.1):
    device = "cuda"
    embedding_size = list(num_to_hidden.values())[0][0].shape[-1]
    logistic_probe_1 = MultiClassLogisticRegression(embedding_size, 8)
    logistic_probe_1 = logistic_probe_1.to(device)
    logistic_probe_2 = MultiClassLogisticRegression(embedding_size, 8)
    logistic_probe_2 = logistic_probe_2.to(device)


    training_x = []
    training_y = []
    training_y_true = []
    for i,j in training_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        training_x.append(x)
        training_y.append(torch.tensor(x_to_y(i,j)//100-2))
        training_y_true.append(torch.tensor((i+j)//100-2))
    training_x = torch.stack(training_x)
    training_y = torch.stack(training_y)
    training_y_true = torch.stack(training_y_true)

    perm = torch.randperm(training_x.shape[0])
    training_x = training_x[perm]
    training_y = training_y[perm]
    training_y_true = training_y_true[perm]
    training_x = training_x.to("cuda")
    training_y = training_y.to("cuda").long()
    training_y_true = training_y_true.to("cuda").long()


    testing_x = []
    testing_y = []
    testing_y_true = []
    for i,j in testing_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        testing_x.append(x)
        testing_y.append(torch.tensor(x_to_y(i,j)//100-2))
        testing_y_true.append(torch.tensor((i+j)//100-2))
    testing_x = torch.stack(testing_x)
    testing_y = torch.stack(testing_y)
    testing_y_true = torch.stack(testing_y_true)
    testing_x = testing_x.to("cuda")
    testing_y = testing_y.to("cuda").long()
    testing_y_true = testing_y_true.to("cuda").long()

    optimizer_1 = torch.optim.Adam(logistic_probe_1.parameters(), lr=lr)
    optimizer_2 = torch.optim.Adam(logistic_probe_2.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        optimizer_1.zero_grad()
        outputs_1 = logistic_probe_1(training_x)
        loss = criterion(outputs_1, training_y)
        loss.backward()
        optimizer_1.step()

        optimizer_2.zero_grad()
        outputs_2 = logistic_probe_2(training_x)
        loss = criterion(outputs_2, training_y_true)
        loss.backward()
        optimizer_2.step()

    logistic_probe_1.eval()
    logistic_probe_2.eval()
    with torch.no_grad():
        #_, y_pred_train = torch.max((logistic_probe(training_x)),1)
        _, y_pred_test = torch.max((logistic_probe_1(testing_x)),1)
        _, y_pred_test_true = torch.max((logistic_probe_2(testing_x)),1)
    
    #y_train_class = (training_y).long()
    y_test_class = (testing_y).long()
    y_test_class_true = (testing_y_true).long()

    #training_correct_predictions = (y_pred_train == y_train_class).float()
    #training_accuracy = training_correct_predictions.mean().item()
    testing_accuracy_1 = (y_pred_test == y_test_class).float().mean().item()
    testing_accuracy_2 = (y_pred_test_true == y_test_class_true).float().mean().item()

    print(f"{testing_accuracy_1=}")
    print(f"{testing_accuracy_2=}")
    
    testing_correct_predictions_all = ((y_pred_test == y_pred_test_true) == (y_test_class == y_test_class_true)).float()
    testing_accuracy_all = testing_correct_predictions_all.mean().item()
    #print(f"{training_accuracy=}")
    print(f"{testing_accuracy_all=}")

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

def get_sum(i,j):
    return i+j

def get_data(i,j):
    return result_dic[(i,j)]

for layer_index in tqdm(range(start_layer, len(num_to_hidden[samples[0]]))):
    #train_linear_probe(num_to_hidden, layer_index, get_sum, samples_train, samples_test, 0.1)
    #train_error_detector(num_to_hidden, layer_index, get_data, samples_train, samples_test, 0.1)
    #train_logistic_probe(num_to_hidden, layer_index, get_data, samples_train, samples_test)
    train_logistic_error_detector(num_to_hidden, layer_index, get_data, samples_train, samples_test)