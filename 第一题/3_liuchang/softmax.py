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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores=X.dot(W)
  for i in range(num_train):
    accum = np.sum(np.exp(scores[i]))   #the denominator of formulation
    for j in range(num_classes):
        if j==y[i]:
            correct = np.exp(scores[i,y[i]])  #the score of correct classifier,看softmax的解释！！
            dW[:,y[i]] -= X[i].T     #其实自己的本意不是这个式子，写错了而已，歪打正着
        dW[:,j] += (X[i].T)*np.exp(scores[i,j])/accum
    loss += -np.log(correct/accum)
  loss/= num_train
  loss += 0.5*reg*np.sum(W*W)
  dW = dW / num_train   #！！！！！
  dW += reg*W
  
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
  #  num_train is dimension
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores=X.dot(W)
  exp_scores = np.exp(scores)
  yemp = np.sum(exp_scores,axis=1)
  temp = -np.log( exp_scores[range(num_train),y] / yemp) #a N×1 array
  yemp = np.reshape(yemp,(num_train,1))   #顺序一颠倒就错，不能用（500,）/（500,1）么？
  loss = np.sum(temp)/num_train
  loss += 0.5*reg*np.sum(W*W)
  
  #exp_scores[np.arange(num_train),y] = 0     
  #放在这儿是警示这句话不能有，与SVM不同，球和的累加项并没有把正确分类的那一项给除掉，所以求导是两部分之和
  dldf = (exp_scores / yemp)  #dldw = dldf*dfdw
  dldf[np.arange(num_train),y]-=1
  dW = (X.T).dot (dldf)
  #for i in range(num_train):
        #dW[:,y[i]] -= X[i].T
  dW = dW / num_train   #！！！！！
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

