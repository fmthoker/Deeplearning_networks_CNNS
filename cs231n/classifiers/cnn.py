import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float64):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C,H,W = input_dim
    self.params['W1'] = weight_scale*np.random.randn(num_filters,C,filter_size,filter_size)
    self.params['b1'] = np.zeros(num_filters)

# outsize of conv_layer 

    stride_conv = 1 # could be different then change value here
    pad = (filter_size - 1) / 2 
    conv_H_out = 1 + ((H + 2 * pad - filter_size) / stride_conv)
    conv_W_out = 1 + ((W + 2 * pad - filter_size) / stride_conv)

    stride_pool = 2 # could be different then change value here
    pool_height = 2
    pool_width = 2

    pool_H_out = 1 + ((conv_H_out  - pool_height) / stride_pool)
    pool_W_out = 1 + ((conv_W_out - pool_width) / stride_pool)

    self.params['W2'] = weight_scale*np.random.randn(num_filters*pool_H_out*pool_W_out,hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale*np.random.randn(hidden_dim,num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out_conv, cache_conv = conv_forward_naive(X,W1,b1,conv_param)
    out_relu, cache_relu = relu_forward(out_conv)
    out_max_pool, cache_maxpool = max_pool_forward_naive(out_relu,pool_param)
    out_hid,cache_hid = affine_relu_forward(out_max_pool,W2,b2)
    scores,cache_output = affine_forward(out_hid,W3,b3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss , dout =  softmax_loss(scores,y)
    r1 = np.sum(self.params['W1']**2)
    r2 = np.sum(self.params['W2']**2)
    r3 = np.sum(self.params['W3']**2)
    loss  += (0.5)*(self.reg)*(r1+r2+r3)  # add regularizer parameter

    dx3,dw3,db3 =  affine_backward(dout,cache_output)
    dx2,dw2,db2 = affine_relu_backward(dx3,cache_hid)
    dmax = max_pool_backward_naive(dx2,cache_maxpool)
    drelu  = relu_backward(dmax,cache_relu)
    dx1,dw1,db1 = conv_backward_naive(drelu,cache_conv)
    
    grads['W1'] = dw1 + (self.reg)*self.params['W1']
    grads['W2'] = dw2 + (self.reg)*self.params['W2']
    grads['W3'] = dw3 + (self.reg)*self.params['W3']
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  

class ThreeConvlayers_convnet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  [conv-relu-pool]X3 - [affine]x2 - [softmax or SVM]
  
  each pool is a 2x2 max pool The network operates on minibatches of data that have shape (N, C, H, W) consisting of N images, each with height H and width W and with C input channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[15,20,20], filter_size=[11,3,3],
               hidden_dim=[100,10], num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float64):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.num_conv_layers = 3
    self.dtype = dtype
    

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    def get_fans(shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
	fan_out = shape[1] if len(shape) == 2 else shape[0]
	return fan_in, fan_out
    C,H,W = input_dim
    stride_conv = 1 # could be different then change value here
    pad1 = (filter_size[0] - 1) / 2 
    pad2 = (filter_size[1] - 1) / 2 
    pad3 = (filter_size[1] - 1) / 2 

    conv_H_out = {}
    conv_W_out = {}
    stride_pool = 2 # could be different then change value here
    pool_height = 2
    pool_width = 2

    conv_H_out[0] = 1 + ((H + 2 * pad1 - filter_size[0]) / stride_conv)
    conv_W_out[0] = 1 + ((W + 2 * pad1 - filter_size[0]) / stride_conv)
    pool_H_out0 = 1 + ((conv_H_out[0]  - pool_height) / stride_pool)
    pool_W_out0 = 1 + ((conv_W_out[0] - pool_width) / stride_pool)

    conv_H_out[1] = 1 + ((pool_H_out0 + 2 * pad2 - filter_size[1]) / stride_conv)
    conv_W_out[1] = 1 + ((pool_W_out0 + 2 * pad2 - filter_size[1]) / stride_conv)
    pool_H_out1 = 1 + ((conv_H_out[1]  - pool_height) / stride_pool)
    pool_W_out1 = 1 + ((conv_W_out[1] - pool_width) / stride_pool)

    conv_H_out[2] = 1 + ((pool_H_out1 + 2 * pad3 - filter_size[2]) / stride_conv)
    conv_W_out[2] = 1 + ((pool_W_out1 + 2 * pad3 - filter_size[2]) / stride_conv)
    pool_H_out2 = 1 + ((conv_H_out[2]  - pool_height) / stride_pool)
    pool_W_out2 = 1 + ((conv_W_out[2] - pool_width) / stride_pool)

    fan_in, fan_out = np.prod((C,filter_size[0],filter_size[0])),num_filters[0]
    self.params['W1'] = np.random.randn(num_filters[0],C,filter_size[0],filter_size[0])*np.sqrt((2/fan_in))
    self.params['b1'] = np.zeros(num_filters[0])

    fan_in, fan_out = np.prod((num_filters[0],filter_size[1],filter_size[1])),num_filters[1]
    self.params['W2'] = np.random.randn(num_filters[1],num_filters[0],filter_size[1],filter_size[1])*np.sqrt((2/fan_in))

    self.params['b2'] = np.zeros(num_filters[1])

    fan_in, fan_out = np.prod((num_filters[1],filter_size[2],filter_size[2])),num_filters[2]
    self.params['W3'] = np.random.randn(num_filters[2],num_filters[1],filter_size[2],filter_size[2])*np.sqrt((2/fan_in))
    self.params['b3'] = np.zeros(num_filters[2])
    # outsize of conv_layer 




    fan_in, fan_out = np.prod((num_filters[2],pool_H_out2,pool_W_out2)),hidden_dim[0]
    self.params['W4'] = np.random.randn(num_filters[2]*pool_H_out2*pool_W_out2,hidden_dim[0])*np.sqrt((2/fan_in))
    self.params['b4'] = np.zeros(hidden_dim[0])

    fan_in, fan_out = hidden_dim[0],hidden_dim[1]
    self.params['W5'] = np.random.randn(hidden_dim[0],hidden_dim[1])*np.sqrt((2/fan_in))
    self.params['b5'] = np.zeros(hidden_dim[1])

    fan_in, fan_out = hidden_dim[1],num_classes
    self.params['W6'] = np.random.randn(hidden_dim[1],num_classes)*np.sqrt((2/fan_in))
    self.params['b6'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    #print W4.shape , b4.shape
    #print W5.shape , b5.shape
    #print W1.shape,W2.shape,W3.shape,W4.shape,W5.shape
    #print b1.shape,b2.shape,b3.shape,b4.shape,b5.shape
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size1 = W1.shape[2]
    filter_size2 = W2.shape[2]
    filter_size3 = W3.shape[2]
    conv_param1 = {'stride': 1, 'pad': (filter_size1 - 1) / 2}
    conv_param2 = {'stride': 1, 'pad': (filter_size2 - 1) / 2}
    conv_param3 = {'stride': 1, 'pad': (filter_size3- 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################

    out_conv1, cache_conv1 = conv_forward_naive(X,W1,b1,conv_param1) #print out_conv.shape
    out_relu1, cache_relu1 = relu_forward(out_conv1) #print out_relu.shape
    output_max_pool1, cache_maxpool1 = max_pool_forward_naive(out_relu1,pool_param) #print layer[i+1].shape

    out_conv2, cache_conv2 = conv_forward_naive(output_max_pool1,W2,b2,conv_param2) #print out_conv.shape
    out_relu2, cache_relu2 = relu_forward(out_conv2) #print out_relu.shape
    output_max_pool2, cache_maxpool2 = max_pool_forward_naive(out_relu2,pool_param) #print layer[i+1].shape

    out_conv3, cache_conv3 = conv_forward_naive(output_max_pool2,W3,b3,conv_param3) #print out_conv.shape
    out_relu3, cache_relu3 = relu_forward(out_conv3) #print out_relu.shape
    output_max_pool3, cache_maxpool3 = max_pool_forward_naive(out_relu3,pool_param) #print layer[i+1].shape

    out_hid1,cache_hid1 = affine_forward(output_max_pool3,W4,b4)
    out_hid2,cache_hid2 = affine_forward(out_hid1,W5,b5)
    scores,cache_output = affine_forward(out_hid2,W6,b6)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss , dout =  softmax_loss(scores,y)
    r1 = np.sum(self.params['W1']**2)
    r2 = np.sum(self.params['W2']**2)
    r3 = np.sum(self.params['W3']**2)
    r4 = np.sum(self.params['W4']**2)
    r5 = np.sum(self.params['W5']**2)
    r6 = np.sum(self.params['W6']**2)
    loss  += (0.5)*(self.reg)*(r1+r2+r3+r4+r5+r6)  # add regularizer parameter

    dx ={}
    dw = {}
    db = {}
    dx6,dw6,db6 =  affine_backward(dout,cache_output)   # gradient of softmax layer
    dx5,dw5,db5 = affine_backward(dx6,cache_hid2)   # gradient of affine layer layer
    dx4,dw4,db4 = affine_backward(dx5,cache_hid1)   # gradient of affine layer layer

    dmax3 = max_pool_backward_naive(dx4,cache_maxpool3)
    drelu3  = relu_backward(dmax3,cache_relu3)
    dx3,dw3,db3 = conv_backward_naive(drelu3,cache_conv3)
    
    dmax2 = max_pool_backward_naive(dx3,cache_maxpool2)
    drelu2  = relu_backward(dmax2,cache_relu2)
    dx2,dw2,db2 = conv_backward_naive(drelu2,cache_conv2)

    dmax1 = max_pool_backward_naive(dx2,cache_maxpool1)
    drelu1  = relu_backward(dmax1,cache_relu1)
    dx1,dw1,db1 = conv_backward_naive(drelu1,cache_conv1)


    grads['W1'] = dw1 + (self.reg)*self.params['W1']
    grads['W2'] = dw2 + (self.reg)*self.params['W2']
    grads['W3'] = dw3 + (self.reg)*self.params['W3']
    grads['W4'] = dw4 + (self.reg)*self.params['W4']
    grads['W5'] = dw5 + (self.reg)*self.params['W5']
    grads['W6'] = dw6 + (self.reg)*self.params['W6']
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    grads['b4'] = db4
    grads['b5'] = db5
    grads['b6'] = db6


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
