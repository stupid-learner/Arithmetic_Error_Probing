import torch
import sys
import os
from model import RidgeRegression, MultiClassLogisticRegression, MLP, CircularProbe, RidgeRegressionErrorDetector
from general_ps_utils import ModelAndTokenizer
from utlis import *
from tqdm import tqdm
from huggingface_hub import login


digit_index = int(sys.argv[1]) #from right to left

if digit_index == 3:
    def edit_for_regression(num):
        return num - 2
    NUM_CLASS = 8
else:
    def edit_for_regression(num):
        return num 
    NUM_CLASS = 10



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
        y_pred_train = get_digit(linear_probe(training_x), digit_index).long()
        y_pred_test = get_digit(linear_probe(testing_x), digit_index).long()
    
    y_train_class = get_digit(training_y, digit_index).long()
    y_test_class = get_digit(testing_y, digit_index).long()

    training_correct_predictions = (y_pred_train == y_train_class).float()
    training_accuracy = training_correct_predictions.mean().item()
    
    testing_correct_predictions = (y_pred_test == y_test_class).float()
    testing_accuracy = testing_correct_predictions.mean().item()
    
    #print(f"{training_accuracy=}")
    #print(f"{testing_accuracy=}")
    return testing_accuracy
    

def train_linear_error_detector(num_to_hidden, layer_index, x_to_y, training_prompt, testing_prompt, lambda_, epochs = 1000, lr = 0.01):
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
        if get_digit((i+j), digit_index) == get_digit(x_to_y(i,j), digit_index):
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
        if get_digit((i+j), digit_index) == get_digit(x_to_y(i,j), digit_index):
            testing_y.append(torch.tensor(1))
        else:
            testing_y.append(torch.tensor(0))
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
    #print(f"{training_accuracy=}")
    #print(f"{testing_accuracy=}")
    return testing_accuracy

def train_linear_error_detector_seperately(num_to_hidden, layer_index, x_to_y, training_prompt, testing_prompt, lambda_, epochs = 1000, lr = 0.01):
    device = "cuda"
    embedding_size = list(num_to_hidden.values())[0][0].shape[-1]
    linear_probe_1 = RidgeRegression(embedding_size, lambda_)
    linear_probe_1 = linear_probe_1.to(device)
    linear_probe_2 = RidgeRegression(embedding_size, lambda_)
    linear_probe_2 = linear_probe_2.to(device)


    training_x = []
    training_y = []
    training_y_true = []
    for i,j in training_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        training_x.append(x)
        training_y.append(torch.tensor(edit_for_regression(get_digit(x_to_y(i,j), digit_index))))
        training_y_true.append(torch.tensor(edit_for_regression(get_digit((i+j), digit_index))))
    training_x = torch.stack(training_x)
    training_y = torch.stack(training_y)
    training_y_true = torch.stack(training_y_true)

    perm = torch.randperm(training_x.shape[0])
    training_x = training_x[perm]
    training_y = training_y[perm]
    training_y_true = training_y_true[perm]
    training_x = training_x.to("cuda")
    training_y = training_y.to("cuda").float()
    training_y_true = training_y_true.to("cuda").float()


    testing_x = []
    testing_y = []
    testing_y_true = []
    for i,j in testing_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        testing_x.append(x)
        testing_y.append(torch.tensor(edit_for_regression(get_digit(x_to_y(i,j), digit_index))))
        testing_y_true.append(torch.tensor(edit_for_regression(get_digit((i+j), digit_index))))
    testing_x = torch.stack(testing_x)
    testing_y = torch.stack(testing_y)
    testing_y_true = torch.stack(testing_y_true)
    testing_x = testing_x.to("cuda")
    testing_y = testing_y.to("cuda").float()
    testing_y_true = testing_y_true.to("cuda").float()

    optimizer_1 = torch.optim.Adam(linear_probe_1.parameters(), lr=lr)
    optimizer_2 = torch.optim.Adam(linear_probe_2.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        optimizer_1.zero_grad()
        outputs_1 = linear_probe_1(training_x)
        loss = criterion(outputs_1, training_y)
        loss.backward()
        optimizer_1.step()

        optimizer_2.zero_grad()
        outputs_2 = linear_probe_2(training_x)
        loss = criterion(outputs_2, training_y_true)
        loss.backward()
        optimizer_2.step()

    linear_probe_1.eval()
    linear_probe_2.eval()
    with torch.no_grad():
        #_, y_pred_train = torch.max((logistic_probe(training_x)),1)
        y_pred_test = torch.round((linear_probe_1(testing_x)))%10
        y_pred_test_true = torch.round((linear_probe_2(testing_x)))%10
    
    #y_train_class = (training_y).long()
    y_test_class = (testing_y).long()
    y_test_class_true = (testing_y_true).long()

    #training_correct_predictions = (y_pred_train == y_train_class).float()
    #training_accuracy = training_correct_predictions.mean().item()
    testing_accuracy_1 = (y_pred_test == y_test_class).float().mean().item()
    testing_accuracy_2 = (y_pred_test_true == y_test_class_true).float().mean().item()

    #print(f"{testing_accuracy_1=}")
    #print(f"{testing_accuracy_2=}")
    
    testing_correct_predictions_all = ((y_pred_test == y_pred_test_true) == (y_test_class == y_test_class_true)).float()
    testing_accuracy_all = testing_correct_predictions_all.mean().item()
    #print(f"{training_accuracy=}")
    #print(f"{testing_accuracy_all=}")
    return testing_accuracy_all


def train_linear_error_detector_jointly(num_to_hidden, layer_index, x_to_y, training_prompt, testing_prompt, lambda_, epochs = 1000, lr = 0.01):
    device = "cuda"
    embedding_size = list(num_to_hidden.values())[0][0].shape[-1]
    linear_probe = RidgeRegressionErrorDetector(embedding_size, lambda_)
    linear_probe = linear_probe.to(device)

    training_x = []
    training_y = []
    for i,j in training_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        training_x.append(x)
        if get_digit((i+j),digit_index) == get_digit(x_to_y(i,j),digit_index):
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
        if get_digit((i+j),digit_index) == get_digit(x_to_y(i,j),digit_index):
            testing_y.append(torch.tensor(1))
        else:
            testing_y.append(torch.tensor(0))
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
    #print(f"{training_accuracy=}")
    #print(f"{testing_accuracy=}")
    return testing_accuracy

def train_logistic_probe(num_to_hidden, layer_index, x_to_y, training_prompt, testing_prompt, epochs = 1000, lr = 0.001):
    device = "cuda"
    embedding_size = list(num_to_hidden.values())[0][0].shape[-1]
    logistic_probe = MultiClassLogisticRegression(embedding_size, NUM_CLASS)
    logistic_probe = logistic_probe.to(device)

    training_x = []
    training_y = []
    for i,j in training_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        training_x.append(x)
        training_y.append(torch.tensor(edit_for_regression(get_digit(x_to_y(i,j),digit_index))))
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
        testing_y.append(torch.tensor(edit_for_regression(get_digit(x_to_y(i,j),digit_index))))
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
    #print(f"{training_accuracy=}")
    #print(f"{testing_accuracy=}")
    return testing_accuracy

def train_logistic_error_detector_seperately(num_to_hidden, layer_index, x_to_y, training_prompt, testing_prompt, epochs = 1000, lr = 0.001):
    device = "cuda"
    embedding_size = list(num_to_hidden.values())[0][0].shape[-1]
    logistic_probe_1 = MultiClassLogisticRegression(embedding_size, NUM_CLASS)
    logistic_probe_1 = logistic_probe_1.to(device)
    logistic_probe_2 = MultiClassLogisticRegression(embedding_size, NUM_CLASS)
    logistic_probe_2 = logistic_probe_2.to(device)


    training_x = []
    training_y = []
    training_y_true = []
    for i,j in training_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        training_x.append(x)
        training_y.append(torch.tensor(edit_for_regression(get_digit(x_to_y(i,j),digit_index))))
        training_y_true.append(torch.tensor(edit_for_regression(get_digit((i+j),digit_index))))
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
        testing_y.append(torch.tensor(edit_for_regression(get_digit(x_to_y(i,j),digit_index))))
        testing_y_true.append(torch.tensor(edit_for_regression(get_digit((i+j),digit_index))))
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

    #print(f"{testing_accuracy_1=}")
    #print(f"{testing_accuracy_2=}")
    
    testing_correct_predictions_all = ((y_pred_test == y_pred_test_true) == (y_test_class == y_test_class_true)).float()
    testing_accuracy_all = testing_correct_predictions_all.mean().item()
    #print(f"{training_accuracy=}")
    #print(f"{testing_accuracy_all=}")
    return testing_accuracy_all

def train_mlp_probe(num_to_hidden, layer_index, x_to_y, training_prompt, testing_prompt, epochs = 1000, lr = 0.001):
    device = "cuda"
    embedding_size = list(num_to_hidden.values())[0][0].shape[-1]
    mlp_probe = MLP(embedding_size, 512, NUM_CLASS)
    mlp_probe = mlp_probe.to(device)

    training_x = []
    training_y = []
    for i,j in training_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        training_x.append(x)
        training_y.append(torch.tensor(edit_for_regression(get_digit(x_to_y(i,j),digit_index))))
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
        testing_y.append(torch.tensor(edit_for_regression(get_digit(x_to_y(i,j),digit_index))))
    testing_x = torch.stack(testing_x)
    testing_y = torch.stack(testing_y)
    testing_x = testing_x.to("cuda")
    testing_y = testing_y.to("cuda").long()

    optimizer = torch.optim.Adam(mlp_probe.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = mlp_probe(training_x)
        loss = criterion(outputs, training_y)
        loss.backward()
        optimizer.step()

    mlp_probe.eval()
    with torch.no_grad():
        _, y_pred_train = torch.max((mlp_probe(training_x)),1)
        _, y_pred_test = torch.max((mlp_probe(testing_x)),1)
    
    y_train_class = (training_y).long()
    y_test_class = (testing_y).long()

    training_correct_predictions = (y_pred_train == y_train_class).float()
    training_accuracy = training_correct_predictions.mean().item()
    
    testing_correct_predictions = (y_pred_test == y_test_class).float()
    testing_accuracy = testing_correct_predictions.mean().item()
    #print(f"{training_accuracy=}")
    #print(f"{testing_accuracy=}")
    return testing_accuracy

def train_mlp_error_detector(num_to_hidden, layer_index, x_to_y, training_prompt, testing_prompt, epochs = 1000, lr = 0.001):
    device = "cuda"
    embedding_size = list(num_to_hidden.values())[0][0].shape[-1]
    mlp_probe = MLP(embedding_size, 512, 2)
    mlp_probe = mlp_probe.to(device)

    training_x = []
    training_y = []
    for i,j in training_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        training_x.append(x)
        if get_digit((i+j),digit_index) == get_digit(x_to_y(i,j),digit_index):
            training_y.append(torch.tensor(1))
        else:
            training_y.append(torch.tensor(0))
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
        if get_digit((i+j),digit_index) == get_digit(x_to_y(i,j),digit_index):
            testing_y.append(torch.tensor(1))
        else:
            testing_y.append(torch.tensor(0))
    testing_x = torch.stack(testing_x)
    testing_y = torch.stack(testing_y)
    testing_x = testing_x.to("cuda")
    testing_y = testing_y.to("cuda").long()

    assert training_x.shape[0] == training_y.shape[0], f"{training_x.shape=}, {training_y.shape=}"
    assert testing_x.shape[0] == testing_y.shape[0], f"{testing_x.shape=}, {testing_y.shape=}"

    optimizer = torch.optim.Adam(mlp_probe.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = mlp_probe(training_x)
        loss = criterion(outputs, training_y)
        loss.backward()
        optimizer.step()

    mlp_probe.eval()
    with torch.no_grad():
        _, y_pred_train = torch.max((mlp_probe(training_x)),1)
        _, y_pred_test = torch.max((mlp_probe(testing_x)),1)
    
    y_train_class = (training_y).long()
    y_test_class = (testing_y).long()

    training_correct_predictions = (y_pred_train == y_train_class).float()
    training_accuracy = training_correct_predictions.mean().item()
    
    testing_correct_predictions = (y_pred_test == y_test_class).float()
    testing_accuracy = testing_correct_predictions.mean().item()
    #print(f"{training_accuracy=}")
    #print(f"{testing_accuracy=}")
    return testing_accuracy

def train_mlp_error_detector_seperately(num_to_hidden, layer_index, x_to_y, training_prompt, testing_prompt, epochs = 1000, lr = 0.001):
    device = "cuda"
    embedding_size = list(num_to_hidden.values())[0][0].shape[-1]
    mlp_probe_1 = MLP(embedding_size, 512, NUM_CLASS)
    mlp_probe_1 = mlp_probe_1.to(device)
    mlp_probe_2 = MLP(embedding_size, 512, NUM_CLASS)
    mlp_probe_2 = mlp_probe_2.to(device)


    training_x = []
    training_y = []
    training_y_true = []
    for i,j in training_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        training_x.append(x)
        training_y.append(torch.tensor(edit_for_regression(get_digit(x_to_y(i,j),digit_index))))
        training_y_true.append(torch.tensor(edit_for_regression(get_digit((i+j),digit_index))))
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
        testing_y.append(torch.tensor(edit_for_regression(get_digit(x_to_y(i,j),digit_index))))
        testing_y_true.append(torch.tensor(edit_for_regression(get_digit((i+j),digit_index))))
    testing_x = torch.stack(testing_x)
    testing_y = torch.stack(testing_y)
    testing_y_true = torch.stack(testing_y_true)
    testing_x = testing_x.to("cuda")
    testing_y = testing_y.to("cuda").long()
    testing_y_true = testing_y_true.to("cuda").long()

    optimizer_1 = torch.optim.Adam(mlp_probe_1.parameters(), lr=lr)
    optimizer_2 = torch.optim.Adam(mlp_probe_2.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        optimizer_1.zero_grad()
        outputs_1 = mlp_probe_1(training_x)
        loss = criterion(outputs_1, training_y)
        loss.backward()
        optimizer_1.step()

        optimizer_2.zero_grad()
        outputs_2 = mlp_probe_2(training_x)
        loss = criterion(outputs_2, training_y_true)
        loss.backward()
        optimizer_2.step()

    mlp_probe_1.eval()
    mlp_probe_2.eval()
    with torch.no_grad():
        #_, y_pred_train = torch.max((mlp_probe(training_x)),1)
        _, y_pred_test = torch.max((mlp_probe_1(testing_x)),1)
        _, y_pred_test_true = torch.max((mlp_probe_2(testing_x)),1)
    
    #y_train_class = (training_y).long()
    y_test_class = (testing_y).long()
    y_test_class_true = (testing_y_true).long()

    #training_correct_predictions = (y_pred_train == y_train_class).float()
    #training_accuracy = training_correct_predictions.mean().item()
    testing_accuracy_1 = (y_pred_test == y_test_class).float().mean().item()
    testing_accuracy_2 = (y_pred_test_true == y_test_class_true).float().mean().item()

    #print(f"{testing_accuracy_1=}")
    #print(f"{testing_accuracy_2=}")
    
    testing_correct_predictions_all = ((y_pred_test == y_pred_test_true) == (y_test_class == y_test_class_true)).float()
    testing_accuracy_all = testing_correct_predictions_all.mean().item()
    #print(f"{training_accuracy=}")
    #print(f"{testing_accuracy_all=}")
    return testing_accuracy_all


def train_circular_probe(num_to_hidden, layer_index, x_to_y, training_prompt, testing_prompt, epochs = 1000, lr = 0.001):
    device = "cuda"
    embedding_size = list(num_to_hidden.values())[0][0].shape[-1]
    circular_probe = CircularProbe(embedding_size)
    circular_probe = circular_probe.to(device)

    training_x = []
    training_y = []
    for i,j in training_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        training_x.append(x)
        training_y.append(torch.tensor(get_digit(x_to_y(i,j),digit_index)))
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
        testing_y.append(torch.tensor(get_digit(x_to_y(i,j),digit_index)))
    testing_x = torch.stack(testing_x)
    testing_y = torch.stack(testing_y)
    testing_x = testing_x.to("cuda")
    testing_y = testing_y.to("cuda").float()

    optimizer = torch.optim.Adam(circular_probe.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = circular_probe.forward_digit(training_x)
        loss = criterion(outputs, training_y)
        loss.backward()
        optimizer.step()

    circular_probe.eval()
    with torch.no_grad():
        y_pred_train = torch.round(circular_probe.forward_digit(training_x))%10
        y_pred_test = torch.round(circular_probe.forward_digit(testing_x))%10
    
    y_train_class = (training_y).long()
    y_test_class = (testing_y).long()

    training_correct_predictions = (y_pred_train == y_train_class).float()
    training_accuracy = training_correct_predictions.mean().item()
    
    testing_correct_predictions = (y_pred_test == y_test_class).float()
    testing_accuracy = testing_correct_predictions.mean().item()
    #print(f"{training_accuracy=}")
    #print(f"{testing_accuracy=}")
    return circular_probe,testing_accuracy

def train_circular_error_detector_seperately(num_to_hidden, layer_index, x_to_y, training_prompt, testing_prompt, epochs = 1000, lr = 0.001):
    device = "cuda"
    embedding_size = list(num_to_hidden.values())[0][0].shape[-1]
    circular_probe_1 = CircularProbe(embedding_size)
    circular_probe_1 = circular_probe_1.to(device)
    circular_probe_2 = CircularProbe(embedding_size)
    circular_probe_2 = circular_probe_2.to(device)


    training_x = []
    training_y = []
    training_y_true = []
    for i,j in training_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        training_x.append(x)
        training_y.append(torch.tensor(get_digit(x_to_y(i,j),digit_index)))
        training_y_true.append(torch.tensor(get_digit((i+j),digit_index)))
    training_x = torch.stack(training_x)
    training_y = torch.stack(training_y)
    training_y_true = torch.stack(training_y_true)

    perm = torch.randperm(training_x.shape[0])
    training_x = training_x[perm]
    training_y = training_y[perm]
    training_y_true = training_y_true[perm]
    training_x = training_x.to("cuda")
    training_y = training_y.to("cuda").float()
    training_y_true = training_y_true.to("cuda").float()


    testing_x = []
    testing_y = []
    testing_y_true = []
    for i,j in testing_prompt:
        hidden_states = num_to_hidden[(i,j)]
        x = hidden_states[layer_index][0][-1]
        testing_x.append(x)
        testing_y.append(torch.tensor(get_digit(x_to_y(i,j),digit_index)))
        testing_y_true.append(torch.tensor(get_digit((i+j),digit_index)))
    testing_x = torch.stack(testing_x)
    testing_y = torch.stack(testing_y)
    testing_y_true = torch.stack(testing_y_true)
    testing_x = testing_x.to("cuda")
    testing_y = testing_y.to("cuda").float()
    testing_y_true = testing_y_true.to("cuda").float()

    optimizer_1 = torch.optim.Adam(circular_probe_1.parameters(), lr=lr)
    optimizer_2 = torch.optim.Adam(circular_probe_2.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        optimizer_1.zero_grad()
        outputs_1 = circular_probe_1.forward_digit(training_x)
        loss = criterion(outputs_1, training_y)
        loss.backward()
        optimizer_1.step()

        optimizer_2.zero_grad()
        outputs_2 = circular_probe_2.forward_digit(training_x)
        loss = criterion(outputs_2, training_y_true)
        loss.backward()
        optimizer_2.step()

    circular_probe_1.eval()
    circular_probe_2.eval()
    with torch.no_grad():
        #_, y_pred_train = torch.max((circular_probe(training_x)),1)
        y_pred_test = torch.round(circular_probe_1.forward_digit(testing_x))%10
        y_pred_test_true = torch.round(circular_probe_2.forward_digit(testing_x))%10
    
    #y_train_class = (training_y).long()
    y_test_class = (testing_y).float()
    y_test_class_true = (testing_y_true).float()

    #training_correct_predictions = (y_pred_train == y_train_class).float()
    #training_accuracy = training_correct_predictions.mean().item()
    testing_accuracy_1 = (y_pred_test == y_test_class).float().mean().item()
    testing_accuracy_2 = (y_pred_test_true == y_test_class_true).float().mean().item()

    #print(f"{testing_accuracy_1=}")
    #print(f"{testing_accuracy_2=}")
    
    testing_correct_predictions_all = ((y_pred_test == y_pred_test_true) == (y_test_class == y_test_class_true)).float()
    testing_accuracy_all = testing_correct_predictions_all.mean().item()
    #print(f"{training_accuracy=}")
    #print(f"{testing_accuracy_all=}")
    return (testing_accuracy_1, testing_accuracy_2, testing_accuracy_all)

torch.manual_seed(42)

if digit_index == 3:
    folder = "arithmetic_data"
elif digit_index == 2:
    folder = "gemma2b_sum_data_fix_digit_3"

result_dic = load_model_result_dic(folder)

samples = []

if digit_index == 3:
    prompt_with_correct_answer = []
    prompt_with_wrong_answer_1 = [] #model_result//100 = correct_answer//100 + 1
    prompt_with_wrong_answer_2 = [] #model_result//100 = correct_answer//100 - 1

    for i in result_dic:
        if i[0]+i[1] == result_dic[i]: 
            prompt_with_correct_answer.append(i)
        elif get_digit((i[0]+i[1]),digit_index) != get_digit(result_dic[i],digit_index): 
            if get_digit(result_dic[i],digit_index) - get_digit((i[0]+i[1]),digit_index) == 1: prompt_with_wrong_answer_1.append(i)
            elif get_digit(result_dic[i],digit_index) - get_digit((i[0]+i[1]),digit_index) == -1: prompt_with_wrong_answer_2.append(i)

    samples += get_balanced_data(prompt_with_correct_answer, 175, digit_index)
    samples += get_balanced_data(prompt_with_wrong_answer_1, 100, digit_index)
    samples += get_balanced_data(prompt_with_wrong_answer_2, 100, digit_index)

elif digit_index == 2:
    prompt_with_correct_answer = []
    prompt_with_wrong_answer = []
    for i in result_dic:
        if i[0]+i[1] == result_dic[i]: 
            prompt_with_correct_answer.append(i)
        elif get_digit((i[0]+i[1]),digit_index) != get_digit(result_dic[i],digit_index):
            prompt_with_wrong_answer.append(i)


    samples += get_balanced_data(prompt_with_correct_answer, 140, digit_index)
    samples += get_balanced_data(prompt_with_wrong_answer, 140, digit_index)

'''
#error_count = 0
for i in result_dic:
    #error_count += 1
    if i[0]+i[1] == result_dic[i]: 
        #error_count -= 1
        prompt_with_correct_answer.append(i)
    elif get_digit((i[0]+i[1]),digit_index) != get_digit(result_dic[i],digit_index): 
        if get_digit(result_dic[i],digit_index) - get_digit((i[0]+i[1]),digit_index) == 1: prompt_with_wrong_answer_1.append(i)
        elif get_digit(result_dic[i],digit_index) - get_digit((i[0]+i[1]),digit_index) == -1: prompt_with_wrong_answer_2.append(i)
#print(f"number of errors is {error_count}")
'''


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

if digit_index == 3:
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
elif digit_index == 2:
    if not os.path.exists("double_circular_probe/num_to_hidden_fix_third_digit"):
        num_to_hidden = {}
        for i in tqdm(samples, delay=120):

            text_for_embeddings = f"{i[0]}+{i[1]}={(i[0]+i[1])//100}"

            x = tokenizer.encode(text_for_embeddings, return_tensors='pt')
            x = x.to(mt.model.device)
            hidden_states = mt.model(x, output_hidden_states=True).hidden_states

            num_to_hidden[i] = hidden_states
        torch.save(num_to_hidden, "double_circular_probe/num_to_hidden_fix_third_digit")
    else:
        num_to_hidden = torch.load("double_circular_probe/num_to_hidden_fix_third_digit")

num_layers = mt.model.config.num_hidden_layers

if len(num_to_hidden[samples[0]]) == num_layers + 1:
    start_layer = 1
else:
    start_layer = 0

def get_sum(i,j):
    return i+j

def get_data(i,j):
    return result_dic[(i,j)]

linear_accuracy = [[],[],[],[],[]]
logistic_accuracy = [[],[],[]]
mlp_accuracy = [[],[],[],[]]
circular_accuracy = [[],[]]

circular_error_2 = []

temp = [[],[],[],[]]

#to add for digit 2: num to hidden, get data
for layer_index in tqdm(range(start_layer, len(num_to_hidden[samples[0]]))):

    circular_error_2.append(train_circular_error_detector_seperately(num_to_hidden, layer_index, get_data, samples_train, samples_test))
    linear_accuracy[0].append(train_linear_probe(num_to_hidden, layer_index, get_sum, samples_train, samples_test, 0.1))    
    logistic_accuracy[0].append(train_logistic_probe(num_to_hidden, layer_index, get_sum, samples_train, samples_test))
    mlp_accuracy[0].append(train_mlp_probe(num_to_hidden, layer_index, get_sum, samples_train, samples_test))

    linear_accuracy[1].append(train_linear_probe(num_to_hidden, layer_index, get_data, samples_train, samples_test, 0.1))
    logistic_accuracy[1].append(train_logistic_probe(num_to_hidden, layer_index, get_data, samples_train, samples_test))
    mlp_accuracy[1].append(train_mlp_probe(num_to_hidden, layer_index, get_data, samples_train, samples_test))

    linear_accuracy[2].append(train_linear_error_detector(num_to_hidden, layer_index, get_data, samples_train, samples_test, 0.1))
    linear_accuracy[3].append(train_linear_error_detector_seperately(num_to_hidden, layer_index, get_data, samples_train, samples_test, 0.1))
    linear_accuracy[4].append(train_linear_error_detector_jointly(num_to_hidden, layer_index, get_data, samples_train, samples_test, 0.1))
    logistic_accuracy[2].append(train_logistic_error_detector_seperately(num_to_hidden, layer_index, get_data, samples_train, samples_test))
    mlp_accuracy[2].append(train_mlp_error_detector_seperately(num_to_hidden, layer_index, get_data, samples_train, samples_test))
    mlp_accuracy[3].append(train_mlp_error_detector(num_to_hidden, layer_index, get_data, samples_train, samples_test))
    
    
    model_1, accuracy_1 = train_circular_probe(num_to_hidden, layer_index, get_sum, samples_train, samples_test)
    model_2, accuracy_2 = train_circular_probe(num_to_hidden, layer_index, get_data, samples_train, samples_test)
    circular_accuracy[0].append((model_1, accuracy_1))
    circular_accuracy[1].append((model_2, accuracy_2))

print("linear:")
for i in linear_accuracy:
    print(max(enumerate(i), key = lambda x: x[1]))

print("logistic:")
for i in logistic_accuracy:
    print(max(enumerate(i), key = lambda x: x[1]))

print("mlp:")
for i in mlp_accuracy:
    print(max(enumerate(i), key = lambda x: x[1]))

print("circular:")
for i in circular_accuracy:
    print(max(enumerate(i), key = lambda x: x[1][1]))


#print(max(circular_error_2, key = lambda x: x[2]))
#torch.save(circular_accuracy, "double_circular_probe/circular_probe_results")