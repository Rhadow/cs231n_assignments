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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  m = X.shape[0]
  for i in range(m):
        x_i = X[i]
        y_i = y[i]
        score = x_i.dot(W)
        stable_score = score - np.max(score)
        exp_sum = 0
        target = 0
        for i, score in enumerate(stable_score):
            exp_score = np.exp(score)
            exp_sum += exp_score
            if i == y_i:
                target = exp_score
        for i, score in enumerate(stable_score):
            exp_score = np.exp(score)
            scale = exp_score / exp_sum
            if i == y_i:
                dW[:, i] += ((scale - 1) * x_i).T
            else:
                dW[:, i] += (scale * x_i).T
        final_score = target / exp_sum
        loss += -np.log(final_score)

  loss /= m
  loss += reg * np.sum(W * W)
  dW /= m
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  m = X.shape[0]

  # Loss Calculation
  logits = X.dot(W)
  stable_logits = logits - np.max(logits, axis=1, keepdims=True)
  scores = np.exp(stable_logits) / np.sum(np.exp(stable_logits), axis=1, keepdims=True)
  loss = np.sum(-np.log(scores[np.arange(m), y]))
  loss /= m
  loss += reg * np.sum(W * W)
  
  # Gradient Calculation
  scores[np.arange(m), y] -= 1
  dW = X.T.dot(scores)
  dW /= m
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

