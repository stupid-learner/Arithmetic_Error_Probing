import torch
import numpy as np

class CircularProbe(torch.nn.Module):
    def __init__(self, embedding_size, basis=10, bias=False):
        super(CircularProbe, self).__init__()
        self.weights = torch.nn.Linear(embedding_size, 2, bias=bias)
        self.basis = basis

    def forward_digit(self, x):
        original_shape = x.shape

        projected =  self.weights(x)

        projected = projected.view(-1, 2)

        # get the angle of the vector
        angles = torch.atan2(projected[...,1], projected[...,0])

        # signed to unsigned
        angles = torch.where(angles < 0, angles + 2*np.pi, angles)
        
        # get the digit
        digits = (angles) * self.basis / (2*np.pi)

        # back to 2d
        res = digits.view(original_shape[:-1])            

        assert res.shape == original_shape[:-1] 
        
        return res
    
    def forward(self, x):
        projected =  self.weights(x)

        return projected

'''
embedding_size = 3
a = CircularProbe(embedding_size, 10, False)
print(a(torch.randn(3,embedding_size)))
print(a.forward_digit(torch.randn(3,embedding_size)))
'''

class CircularErrorDetector(torch.nn.Module):
    def __init__(self, embedding_size, basis, bias):
        super(CircularErrorDetector, self).__init__()
        self.projection_1 = torch.nn.Linear(embedding_size, 2, bias=bias)
        self.projection_2 = torch.nn.Linear(embedding_size, 2, bias=bias)
        self.basis = basis

    def forward(self, x):
        #x: [batch_size, embedding_size]
        projected_1 = self.projection_1(x) #[batch_size, 2]
        projected_2 = self.projection_2(x)

        angle_1 = torch.atan2(projected_1[...,1], projected_1[...,0]) #[batch_size]
        angle_2 = torch.atan2(projected_2[...,1], projected_2[...,0])

        return torch.sigmoid(angle_1 - angle_2)

'''
embedding_size = 3
a = CircularErrorDetector(embedding_size, 10, False)
print(a(torch.randn(3,embedding_size)))
'''


class RidgeRegression(torch.nn.Module):
    def __init__(self, input_dim, lambda_):
        super(RidgeRegression, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(input_dim, requires_grad=True))
        self.b = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        self.lambda_ = lambda_   

    def forward(self, X):
        return X @ self.w + self.b

    def loss(self, X, y):
        mse_loss = torch.mean((self.forward(X) - y) ** 2)
        l2_reg = self.lambda_ * torch.sum(self.w ** 2)  
        return mse_loss + l2_reg

class RidgeRegressionErrorDetector(torch.nn.Module):
    def __init__(self, input_dim, lambda_):
        super(RidgeRegressionErrorDetector, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(input_dim, requires_grad=True))
        self.b1 = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        self.w2 = torch.nn.Parameter(torch.randn(input_dim, requires_grad=True))
        self.b2 = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        self.lambda_ = lambda_   

    def forward(self, X):
        return (X @ self.w1 + self.b1) - (X @ self.w2 + self.b2 )

    def loss(self, X, y):
        mse_loss = torch.mean((self.forward(X) - y) ** 2)
        l2_reg = self.lambda_ * (torch.sum(self.w1 ** 2) + torch.sum(self.w2 ** 2))  
        return mse_loss + l2_reg

class MultiClassLogisticRegression(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiClassLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)


class MLP(torch.nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512, output_dim=8):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class LinearBinaryClassifier(torch.nn.Module):
    def __init__(self, input_dim):
        super(LinearBinaryClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
    
    def forward(self, x):
        out = self.linear(x)
        out = torch.sigmoid(out)  
        return out



# Train circular probe
def train_circular_probe(X, Y, epochs=10000, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Training data Y distribution:")
    for i in range(10):
        count = (y_values == i).sum()
        print(f"Number {i} appears {count} times in training data")
    
    embedding_size = X.shape[-1]
    circular_probe = CircularProbe(embedding_size)
    circular_probe = circular_probe.to(device)

    optimizer = torch.optim.AdamW(circular_probe.parameters(), lr=lr)
    criterion = torch.nn.SmoothL1Loss(beta=1.0)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = circular_probe.forward_digit(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

    return circular_probe

# Test circular probe
def test_probe_circular(X, Y, probe):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Test data Y distribution:")
    for i in range(10):
        count = (y_values == i).sum()
        print(f"Number {i} appears {count} times in test data")
    
    probe = probe.to(device)
    probe.eval()
    with torch.no_grad():
        y_pred = torch.round(probe.forward_digit(X))%10
    correct_predictions = (y_pred == Y).float()
    accuracy = correct_predictions.mean().item()
    return accuracy

# Train ridge regression (linear) probe
def train_ridge_probe(X, Y, lambda_=0.1, epochs=1000, lr=0.1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Training data Y distribution:")
    for i in range(10):
        count = (y_values == i).sum()
        print(f"Number {i} appears {count} times in training data")
    
    embedding_size = X.shape[-1]
    ridge_probe = RidgeRegression(embedding_size, lambda_)
    ridge_probe = ridge_probe.to(device)
    
    Y = Y.float()
    
    optimizer = torch.optim.Adam(ridge_probe.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = ridge_probe.loss(X, Y)
        loss.backward()
        optimizer.step()
    
    return ridge_probe

# Test ridge probe
def test_probe_ridge(X, Y, probe):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Test data Y distribution:")
    for i in range(10):
        count = (y_values == i).sum()
        print(f"Number {i} appears {count} times in test data")
    
    probe = probe.to(device)
    probe.eval()
    with torch.no_grad():
        y_pred = probe(X).long()
        y_test_class = Y.long()
    correct_predictions = (y_pred == y_test_class).float()
    accuracy = correct_predictions.mean().item()
    return accuracy

# Train MLP probe
def train_mlp_probe(X, Y, epochs=1000, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Training data Y distribution:")
    for i in range(10):
        count = (y_values == i).sum()
        print(f"Number {i} appears {count} times in training data")
    
    embedding_size = X.shape[-1]
    num_classes = 10  # For digits 0-9
    mlp_probe = MLP(embedding_size, 512, num_classes)
    mlp_probe = mlp_probe.to(device)
    
    optimizer = torch.optim.Adam(mlp_probe.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = mlp_probe(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
    
    return mlp_probe

# Test MLP probe
def test_probe_mlp(X, Y, probe):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Test data Y distribution:")
    for i in range(10):
        count = (y_values == i).sum()
        print(f"Number {i} appears {count} times in test data")
    
    probe = probe.to(device)
    probe.eval()
    with torch.no_grad():
        _, y_pred = torch.max(probe(X), 1)
    correct_predictions = (y_pred == Y).float()
    accuracy = correct_predictions.mean().item()
    return accuracy

# Train logistic regression probe
def train_logistic_probe(X, Y, epochs=1000, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Training data Y distribution:")
    for i in range(10):
        count = (y_values == i).sum()
        print(f"Number {i} appears {count} times in training data")

    embedding_size = X.shape[-1]
    logistic_probe = MultiClassLogisticRegression(embedding_size, 10)
    logistic_probe = logistic_probe.to(device)

    optimizer = torch.optim.Adam(logistic_probe.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = logistic_probe(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

    return logistic_probe

# Test logistic probe
def test_probe_logistic(X, Y, probe):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Test data Y distribution:")
    for i in range(10):
        count = (y_values == i).sum()
        print(f"Number {i} appears {count} times in test data")
    
    probe = probe.to(device)
    probe.eval()
    with torch.no_grad():
        outputs = probe(X)
        _, y_pred = torch.max(outputs, 1)
    correct_predictions = (y_pred == Y).float()
    accuracy = correct_predictions.mean().item()
    return accuracy

#-----------------------------------------------------------------------------
# Error detector probe training functions
#-----------------------------------------------------------------------------

# Train logistic error detector separately
def train_logistic_error_detector_seperately(X, Y_model, Y_true, epochs=1000, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y_model = Y_model.to(device)
    Y_true = Y_true.to(device)
    
    # Print Y distribution
    y_model_values = Y_model.detach().cpu().numpy()
    y_true_values = Y_true.detach().cpu().numpy()
    print("Training data Y_model and Y_true distributions:")
    for i in range(10):
        count_model = (y_model_values == i).sum()
        count_true = (y_true_values == i).sum()
        print(f"Number {i} appears {count_model} times in Y_model and {count_true} times in Y_true")
    
    embedding_size = X.shape[-1]
    logistic_probe_1 = MultiClassLogisticRegression(embedding_size, 10)
    logistic_probe_1 = logistic_probe_1.to(device)
    logistic_probe_2 = MultiClassLogisticRegression(embedding_size, 10)
    logistic_probe_2 = logistic_probe_2.to(device)
    
    # Train the two probes separately
    optimizer_1 = torch.optim.Adam(logistic_probe_1.parameters(), lr=lr)
    optimizer_2 = torch.optim.Adam(logistic_probe_2.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # Train probe 1 on model outputs
        optimizer_1.zero_grad()
        outputs_1 = logistic_probe_1(X)
        loss_1 = criterion(outputs_1, Y_model)
        loss_1.backward()
        optimizer_1.step()
        
        # Train probe 2 on true answers
        optimizer_2.zero_grad()
        outputs_2 = logistic_probe_2(X)
        loss_2 = criterion(outputs_2, Y_true)
        loss_2.backward()
        optimizer_2.step()
    
    return (logistic_probe_1, logistic_probe_2)



# Train MLP error detector
def train_mlp_error_detector(X, Y, epochs=1000, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Training data Y distribution:")
    for i in range(2):  # Binary classification: 0=wrong, 1=correct
        count = (y_values == i).sum()
        print(f"Class {i} appears {count} times in training data")
    
    embedding_size = X.shape[-1]
    mlp_probe = MLP(embedding_size, 512, 2)  # Binary classification
    mlp_probe = mlp_probe.to(device)
    
    optimizer = torch.optim.Adam(mlp_probe.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = mlp_probe(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
    
    return mlp_probe


# Train MLP error detector separately
def train_mlp_error_detector_seperately(X, Y_model, Y_true, epochs=1000, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y_model = Y_model.to(device)
    Y_true = Y_true.to(device)
    
    # Print Y distribution
    y_model_values = Y_model.detach().cpu().numpy()
    y_true_values = Y_true.detach().cpu().numpy()
    print("Training data Y_model and Y_true distributions:")
    for i in range(10):
        count_model = (y_model_values == i).sum()
        count_true = (y_true_values == i).sum()
        print(f"Number {i} appears {count_model} times in Y_model and {count_true} times in Y_true")
    
    embedding_size = X.shape[-1]
    mlp_probe_1 = MLP(embedding_size, 512, 10)
    mlp_probe_1 = mlp_probe_1.to(device)
    mlp_probe_2 = MLP(embedding_size, 512, 10)
    mlp_probe_2 = mlp_probe_2.to(device)
    
    # Train the two probes separately
    optimizer_1 = torch.optim.Adam(mlp_probe_1.parameters(), lr=lr)
    optimizer_2 = torch.optim.Adam(mlp_probe_2.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # Train probe 1 on model outputs
        optimizer_1.zero_grad()
        outputs_1 = mlp_probe_1(X)
        loss_1 = criterion(outputs_1, Y_model)
        loss_1.backward()
        optimizer_1.step()
        
        # Train probe 2 on true answers
        optimizer_2.zero_grad()
        outputs_2 = mlp_probe_2(X)
        loss_2 = criterion(outputs_2, Y_true)
        loss_2.backward()
        optimizer_2.step()
    
    return (mlp_probe_1, mlp_probe_2)

# Train circular error detector separately
def train_circular_error_detector_seperately(X, Y_model, Y_true, epochs=10000, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y_model = Y_model.to(device)
    Y_true = Y_true.to(device)
    
    # Print Y distribution
    y_model_values = Y_model.detach().cpu().numpy()
    y_true_values = Y_true.detach().cpu().numpy()
    print("Training data Y_model and Y_true distributions:")
    for i in range(10):
        count_model = (y_model_values == i).sum()
        count_true = (y_true_values == i).sum()
        print(f"Number {i} appears {count_model} times in Y_model and {count_true} times in Y_true")
    
    embedding_size = X.shape[-1]
    circular_probe_1 = CircularProbe(embedding_size)
    circular_probe_1 = circular_probe_1.to(device)
    circular_probe_2 = CircularProbe(embedding_size)
    circular_probe_2 = circular_probe_2.to(device)
    
    # Train the two probes separately
    optimizer_1 = torch.optim.Adam(circular_probe_1.parameters(), lr=lr)
    optimizer_2 = torch.optim.Adam(circular_probe_2.parameters(), lr=lr)
    criterion = torch.nn.SmoothL1Loss(beta=1.0)
    
    for epoch in range(epochs):
        # Train probe 1 on model outputs
        optimizer_1.zero_grad()
        outputs_1 = circular_probe_1.forward_digit(X)
        loss_1 = criterion(outputs_1, Y_model.float())
        loss_1.backward()
        optimizer_1.step()
        
        # Train probe 2 on true answers
        optimizer_2.zero_grad()
        outputs_2 = circular_probe_2.forward_digit(X)
        loss_2 = criterion(outputs_2, Y_true.float())
        loss_2.backward()
        optimizer_2.step()
    
    return (circular_probe_1, circular_probe_2)


# Train circular error detector jointly
def train_circular_error_detector_jointly(X, Y, epochs=10000, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Training data Y distribution:")
    for i in range(2):  # Binary classification: 0=wrong, 1=correct
        count = (y_values == i).sum()
        print(f"Class {i} appears {count} times in training data")
    
    embedding_size = X.shape[-1]
    circular_probe = CircularErrorDetector(embedding_size, 10, False)
    circular_probe = circular_probe.to(device)
    
    optimizer = torch.optim.Adam(circular_probe.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = circular_probe(X)
        loss = criterion(outputs, Y.float())
        loss.backward()
        optimizer.step()
    
    return circular_probe

# Test logistic error detector separately
def test_logistic_error_detector_seperately(X, Y_model, Y_true, probes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y_model = Y_model.to(device)
    Y_true = Y_true.to(device)
    
    # Print Y distribution
    y_model_values = Y_model.detach().cpu().numpy()
    y_true_values = Y_true.detach().cpu().numpy()
    print("Test data Y_model and Y_true distributions:")
    for i in range(10):
        count_model = (y_model_values == i).sum()
        count_true = (y_true_values == i).sum()
        print(f"Number {i} appears {count_model} times in Y_model and {count_true} times in Y_true")
    
    probe_1, probe_2 = probes
    probe_1 = probe_1.to(device)
    probe_2 = probe_2.to(device)
    
    probe_1.eval()
    probe_2.eval()
    
    with torch.no_grad():
        _, pred_model = torch.max(probe_1(X), 1)
        _, pred_true = torch.max(probe_2(X), 1)
    
    # Check if the prediction about whether model matches ground truth is correct
    model_correct = (Y_model == Y_true)
    pred_correct = (pred_model == pred_true)
    
    # Calculate metrics
    TP = ((pred_correct == 1) & (model_correct == 1)).float().sum().item()
    FP = ((pred_correct == 1) & (model_correct == 0)).float().sum().item()
    TN = ((pred_correct == 0) & (model_correct == 0)).float().sum().item()
    FN = ((pred_correct == 0) & (model_correct == 1)).float().sum().item()
    
    accuracy = (model_correct == pred_correct).float().mean().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    return accuracy, precision, recall, f1

# Test MLP error detector
def test_mlp_error_detector(X, Y, probe):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Test data Y distribution:")
    for i in range(2):  # Binary classification: 0=wrong, 1=correct
        count = (y_values == i).sum()
        print(f"Class {i} appears {count} times in test data")
    
    probe = probe.to(device)
    probe.eval()
    
    with torch.no_grad():
        _, y_pred = torch.max(probe(X), 1)
    
    # Calculate metrics - treating 1 (correct) as positive class
    TP = ((y_pred == 1) & (Y == 1)).float().sum().item()
    FP = ((y_pred == 1) & (Y == 0)).float().sum().item()
    TN = ((y_pred == 0) & (Y == 0)).float().sum().item()
    FN = ((y_pred == 0) & (Y == 1)).float().sum().item()
    
    correct_predictions = (y_pred == Y).float()
    accuracy = correct_predictions.mean().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    return accuracy, precision, recall, f1

# Test MLP error detector separately
def test_mlp_error_detector_seperately(X, Y_model, Y_true, probes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y_model = Y_model.to(device)
    Y_true = Y_true.to(device)
    
    # Print Y distribution
    y_model_values = Y_model.detach().cpu().numpy()
    y_true_values = Y_true.detach().cpu().numpy()
    print("Test data Y_model and Y_true distributions:")
    for i in range(10):
        count_model = (y_model_values == i).sum()
        count_true = (y_true_values == i).sum()
        print(f"Number {i} appears {count_model} times in Y_model and {count_true} times in Y_true")
    
    probe_1, probe_2 = probes
    probe_1 = probe_1.to(device)
    probe_2 = probe_2.to(device)
    
    probe_1.eval()
    probe_2.eval()
    
    with torch.no_grad():
        _, pred_model = torch.max(probe_1(X), 1)
        _, pred_true = torch.max(probe_2(X), 1)
    
    # Check if the prediction about whether model matches ground truth is correct
    model_correct = (Y_model == Y_true)
    pred_correct = (pred_model == pred_true)
    
    # Calculate metrics
    TP = ((pred_correct == 1) & (model_correct == 1)).float().sum().item()
    FP = ((pred_correct == 1) & (model_correct == 0)).float().sum().item()
    TN = ((pred_correct == 0) & (model_correct == 0)).float().sum().item()
    FN = ((pred_correct == 0) & (model_correct == 1)).float().sum().item()
    
    accuracy = (model_correct == pred_correct).float().mean().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    return accuracy, precision, recall, f1

# Test circular error detector separately
def test_circular_error_detector_seperately(X, Y_model, Y_true, probes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y_model = Y_model.to(device)
    Y_true = Y_true.to(device)
    
    # Print Y distribution
    y_model_values = Y_model.detach().cpu().numpy()
    y_true_values = Y_true.detach().cpu().numpy()
    print("Test data Y_model and Y_true distributions:")
    for i in range(10):
        count_model = (y_model_values == i).sum()
        count_true = (y_true_values == i).sum()
        print(f"Number {i} appears {count_model} times in Y_model and {count_true} times in Y_true")
    
    probe_1, probe_2 = probes
    probe_1 = probe_1.to(device)
    probe_2 = probe_2.to(device)
    
    probe_1.eval()
    probe_2.eval()
    
    with torch.no_grad():
        pred_model = torch.round(probe_1.forward_digit(X)) % 10
        pred_true = torch.round(probe_2.forward_digit(X)) % 10
    
    # Check if the prediction about whether model matches ground truth is correct
    model_correct = (Y_model == Y_true)
    pred_correct = (pred_model == pred_true)
    
    # Calculate metrics
    TP = ((pred_correct == 1) & (model_correct == 1)).float().sum().item()
    FP = ((pred_correct == 1) & (model_correct == 0)).float().sum().item()
    TN = ((pred_correct == 0) & (model_correct == 0)).float().sum().item()
    FN = ((pred_correct == 0) & (model_correct == 1)).float().sum().item()
    
    accuracy = (model_correct == pred_correct).float().mean().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    return accuracy, precision, recall, f1

# Test circular error detector jointly
def test_circular_error_detector_jointly(X, Y, probe):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Test data Y distribution:")
    for i in range(2):  # Binary classification: 0=wrong, 1=correct
        count = (y_values == i).sum()
        print(f"Class {i} appears {count} times in test data")
    
    probe = probe.to(device)
    probe.eval()
    
    with torch.no_grad():
        predicted_result = probe(X)
        predicted_labels = (predicted_result > 0.5).int()
    
    # Calculate metrics - treating 1 (correct) as positive class
    TP = ((predicted_labels == 1) & (Y == 1)).float().sum().item()
    FP = ((predicted_labels == 1) & (Y == 0)).float().sum().item()
    TN = ((predicted_labels == 0) & (Y == 0)).float().sum().item()
    FN = ((predicted_labels == 0) & (Y == 1)).float().sum().item()
    
    correct_predictions = (predicted_labels == Y).float()
    accuracy = correct_predictions.mean().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    return accuracy, precision, recall, f1