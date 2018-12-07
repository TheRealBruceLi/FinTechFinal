# Bruce Li, Evan Yukevich, and Preston Vander Vos
# Carnegie Mellon University
# 70-339 FinTech
# Final Project

# Source: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# Lots of code was taken from the above tutorial

import torch
from data import *

torch.manual_seed(123)

(data, label) = preprocessData("data/")
(dataN, labelN) = normalizeData(data, label)
(trainDataN, trainLabelN, validateDataN, validateLabelN) = splitData(dataN, labelN)
# (trainDataN1, validateDataN1) = holdConstantData(trainDataN, validateDataN, 0)

# D_in is input dimension; H is hidden dimension; D_out is output dimension.
D_in, H, D_out = trainDataN.shape[1], 1, trainLabelN.shape[1]
epochs = 15001

trainX = torch.from_numpy(trainDataN).float()
trainY = torch.from_numpy(trainLabelN).float()
validateX = torch.from_numpy(validateDataN).float()
validateY = torch.from_numpy(validateLabelN).float()

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Sigmoid(),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 5e-6
for t in range(epochs):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(trainX)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, trainY)
    if(t % 250 == 0):
        y_predValidate = model(validateX)
        lossValidate = loss_fn(y_predValidate, validateY)
        print("Epoch: %i Train Loss: %d Validate Loss: %d" % (t, loss.item(), lossValidate.item()))

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
