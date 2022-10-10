"""
Student Name:
Student ID:
Assignment #: 1
Student Email:
Course Name: 5012 Machine Learning
"""

"""
           Programming Assignment 1
         

This program learns a softmax model for the Iris dataset (included).
There is a function, compute_softmax_loss, that computes the
softmax loss and the gradient. It is left empty. Your task is to write
the function.
     
"""
import numpy as np
import os
import math

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

def get_data():
    # Load datasets.
    train_data = np.genfromtxt(IRIS_TRAINING, skip_header=1, 
        dtype=float, delimiter=',') 
    test_data = np.genfromtxt(IRIS_TEST, skip_header=1, 
        dtype=float, delimiter=',') 
    train_x = train_data[:, :4]
    train_y = train_data[:, 4].astype(np.int64)
    test_x = test_data[:, :4]
    test_y = test_data[:, 4].astype(np.int64)

    return train_x, train_y, test_x, test_y

def compute_softmax_loss(W, X, y, reg):
    """
    Softmax loss function.
    Inputs:
    - W: D x K array of weight, where K is the number of classes.
    - X: N x D array of training data. Each row is a D-dimensional point.
    - y: 1-d array of shape (N, ) for the training labels.
    - reg: weight regularization coefficient.

    Returns:
    - softmax loss: NLL/N +  0.5 *reg* L2 regularization,
            
    - dW: the gradient for W.
    """


    #############################################################################
    # TODO: Compute the softmax loss and its gradient.                          #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        f = X[i].dot(W)  # f is wx
        f -= np.max(f)   # used to avoid numeric instability
        correct_f = f[y[i]]
        denom = np.sum(np.exp(f))  # denominator
        p = np.exp(correct_f) / denom
        loss += -np.log(p)
        for j in range(num_classes):
            if j == y[i]:
                dW[:, y[i]] += (np.exp(f[j]) / denom - 1) * X[i, :]
            else:
                dW[:, j] += (np.exp(f[j]) / denom) * X[i, :]
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

def predict(W, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - W: D x K array of weights. K is the number of classes.
    - X: N x D array of training data. Each row is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    y_pred = np.zeros(X.shape[0])
    y_pred = np.argmax(X.dot(W), axis = 1)
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred

def acc(ylabel, y_pred):
    return np.mean(ylabel == y_pred)


def train(X, y, Xtest, ytest, learning_rate=1e-3, reg=1e-5, epochs=100, batch_size=20):
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    num_iters_per_epoch = int(math.floor(1.0*num_train/batch_size))
    
    # randomly initialize W
    W = 0.001 * np.random.randn(dim, num_classes)

    for epoch in range(max_epochs):
        perm_idx = np.random.permutation(num_train)
        # perform mini-batch SGD update
        for it in range(num_iters_per_epoch):
            idx = perm_idx[it*batch_size:(it+1)*batch_size]
            batch_x = X[idx]
            batch_y = y[idx]
            
            # evaluate loss and gradient
            loss, grad = compute_softmax_loss(W, batch_x, batch_y, reg)

            # update parameters
            W += -learning_rate * grad
            

        # evaluate and print every 10 steps
        if epoch % 10 == 0:
            train_acc = acc(y, predict(W, X))
            test_acc = acc(ytest, predict(W, Xtest))
            print('Epoch %4d: loss = %.2f, train_acc = %.4f, test_acc = %.4f' \
                % (epoch, loss, train_acc, test_acc))
    
    return W

max_epochs = 200
batch_size = 20
learning_rate = 0.1
reg = 0.01

# get training and testing data
train_x, train_y, test_x, test_y = get_data()
W = train(train_x, train_y, test_x, test_y, learning_rate, reg, max_epochs, batch_size)

# Classify two new flower samples.
def new_samples():
    return np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
new_x = new_samples()
predictions = predict(W, new_x)

print("New Samples, Class Predictions:    {}\n".format(predictions))

