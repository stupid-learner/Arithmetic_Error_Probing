import torch
import numpy as np

class CircularProbe(torch.nn.Module):
    def __init__(self, embedding_size, basis, bias):
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
    
class MultiClassLogisticRegression(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiClassLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)
