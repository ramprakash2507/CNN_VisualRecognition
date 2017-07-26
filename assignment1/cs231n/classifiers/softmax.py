import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  exp_scores = np.exp(np.dot(X,W))
  for i in range(num_train):
    loss = loss - np.log(exp_scores[i][y[i]]/np.sum(exp_scores[i]))
    X_temp = np.dot(np.resize(X[i], (X[i].shape[0], 1)),np.resize(exp_scores[i],(1,exp_scores[i].shape[0])))/np.sum(exp_scores[i])
    dW = dW + X_temp#np.resize(X_temp, (X_temp.shape[0], 1))
    dW[:, y[i]] -= X[i] 
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss = loss/num_train
  loss = loss + 0.5*reg*np.sum(W*W)
  dW = dW/num_train
  dW = dW + reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  exp_scores = np.exp(np.dot(X,W))
  loss = - np.sum(np.log(exp_scores[range(exp_scores.shape[0]), y]/np.sum(exp_scores, axis=1)))
  dW = np.dot(np.transpose(X),exp_scores/np.resize(np.sum(exp_scores, axis = 1), (exp_scores.shape[0], 1)))
  y_map = np.zeros(exp_scores.shape)
  y_map[range(y_map.shape[0]), y] = 1
  dW = dW - np.dot(np.transpose(X), y_map)
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss = loss/X.shape[0]
  loss = loss + 0.5*reg*np.sum(W*W)
  dW = dW/X.shape[0]
  dW = dW + reg*W
  return loss, dW

