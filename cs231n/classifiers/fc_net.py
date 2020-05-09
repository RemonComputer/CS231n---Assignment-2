from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        h1, h1_cache = affine_forward(X, self.params['W1'], self.params['b1'])
        z1, z1_cache = relu_forward(h1)
        scores, scores_cache = affine_forward(z1, self.params['W2'], self.params['b2'])
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        loss, dout = softmax_loss(scores, y)
        dout, grads['W2'], grads['b2'] = affine_backward(dout, scores_cache)
        dout = relu_backward(dout, z1_cache)
        _, grads['W1'], grads['b1'] = affine_backward(dout, h1_cache)
        
        # regularization effect
        loss += 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2))
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dim[0]))
        # self.params['b1'] = np.zeros(hidden_dim[0])
        
        #print('Initial settings')
        all_dims = [input_dim] + hidden_dims[:] + [num_classes]
        #print('all_dims:', all_dims)
        for layer_idx in range(1, len(all_dims)):
          layer_number = layer_idx
          #print('layer_idx:', layer_idx)
          layer_input_dim = all_dims[layer_idx - 1]
          layer_output_dim =  all_dims[layer_idx]
          affine_layer_weight_name = 'W{}'.format(layer_number)
          affine_layer_bias_name = 'b{}'.format(layer_number)
          #print('affine_layer_weight_name:', affine_layer_weight_name)
          #print('affine_layer_bias_name', affine_layer_bias_name)
          W_i = np.random.normal(loc=0.0, scale=weight_scale, size=(layer_input_dim, layer_output_dim))
          b_i = np.zeros(layer_output_dim)
          #print('W_i.shape:',W_i.shape)
          #print('b_i.shape', b_i.shape) 
          self.params[affine_layer_weight_name] = W_i  
          self.params[affine_layer_bias_name] = b_i
          if self.normalization is not None and layer_idx < (len(all_dims) - 1):
             self.params['gamma{}'.format(layer_number)] = np.ones(layer_output_dim)
             self.params['beta{}'.format(layer_number)] = np.zeros(layer_output_dim)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        cashes = []
        input_i = X
        #print('len(self.bn_params):', len(self.bn_params))
        for layer_idx in range(self.num_layers - 1):
          layer_number = layer_idx + 1
          W_i = self.params['W{}'.format(layer_number)]
          b_i = self.params['b{}'.format(layer_number)]
          h_i, h_i_cache = affine_forward(input_i, W_i, b_i)
          cashes.append(h_i_cache)
          if self.normalization is not None:
            gamma_i_name = 'gamma{}'.format(layer_number)
            beta_i_name = 'beta{}'.format(layer_number)
            #print('gamma_i_name:', gamma_i_name)
            #print('beta_i_name:', beta_i_name)
            gamma_i = self.params[gamma_i_name]
            beta_i = self.params[beta_i_name]
            if self.normalization == 'batchnorm':
              h_i, h_i_cache = batchnorm_forward(h_i, gamma_i, beta_i, self.bn_params[layer_idx])
            elif self.normalization == 'layernorm':
              h_i, h_i_cache = layernorm_forward(h_i, gamma_i, beta_i, self.bn_params[layer_idx])
            cashes.append(h_i_cache)
          z_i, z_i_cache = relu_forward(h_i)
          cashes.append(z_i_cache)
          input_i = z_i
        
        # output layer calculation
        layer_number = self.num_layers
        W_i = self.params['W{}'.format(layer_number)]
        b_i = self.params['b{}'.format(layer_number)]
        scores, scores_cache = affine_forward(input_i, W_i, b_i)
        cashes.append(scores_cache)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # calculating the loss and backward upstream gradient from the loss function
        loss, dout = softmax_loss(scores, y)
        #print('intial softmax loss: ', loss)
        #print('inital dout:', dout)

        layer_idx = self.num_layers - 1
        cache_idx = len(cashes) - 1
        
        # performing some pre-calculations
        layer_number = layer_idx + 1
        affine_weight_name = 'W{}'.format(layer_number)
        affine_bias_name = 'b{}'.format(layer_number)  
        W_i = self.params[affine_weight_name]
        b_i = self.params[affine_bias_name]
        

        # updating the loss for the regularization part
        loss += 0.5 * self.reg * (np.sum(W_i ** 2))
        
        # calculating the output layer parameters gradient
        layer_cache = cashes[cache_idx]
        cache_idx -= 1
        dout, grad_W_i, grad_b_i = affine_backward(dout, layer_cache)

        # perfomring the gradient with respect to regularization
        grad_W_i += self.reg * W_i

        # setting the calculations in the grad dict
        grads[affine_weight_name] = grad_W_i
        grads[affine_bias_name] = grad_b_i
        layer_idx -= 1

        #print('begin backward pass')
        while cache_idx >= 0:
          # performing some pre-calculations
          layer_number = layer_idx + 1
          affine_weight_name = 'W{}'.format(layer_number)
          affine_bias_name = 'b{}'.format(layer_number)  
          W_i = self.params[affine_weight_name]
          b_i = self.params[affine_bias_name]
          # updating the loss for the regularization part
          loss += 0.5 * self.reg * (np.sum(W_i ** 2))
          # calculating the relu_i layer backward pass
          layer_cache = cashes[cache_idx]
          cache_idx -= 1
          dout = relu_backward(dout, layer_cache)
          if self.normalization is not None :
            layer_cache = cashes[cache_idx]
            cache_idx -= 1
            if self.normalization == 'batchnorm':
              dout, dgamma, dbeta = batchnorm_backward_alt(dout, layer_cache)
            elif self.normalization == 'layernorm':
              dout, dgamma, dbeta = layernorm_backward(dout, layer_cache)
            gamma_i_name = 'gamma{}'.format(layer_number)
            beta_i_name = 'beta{}'.format(layer_number)
            grads[gamma_i_name] = dgamma
            grads[beta_i_name] = dbeta
            #print('gamma_i_name:', gamma_i_name)
            #print('beta_i_name:', beta_i_name)
            #print('grads[gamma_i_name].shape:', grads[gamma_i_name].shape)
            #print('grads[beta_i_name]:', grads[beta_i_name].shape)
            
            
          # calculating the  affine layer backward
          layer_cache = cashes[cache_idx]
          cache_idx -= 1
          dout, grad_W_i, grad_b_i = affine_backward(dout, layer_cache)
          # perfomring the gradient with respect to regularization
          grad_W_i += self.reg * W_i
          # setting the calculations in the grad dict
          grads[affine_weight_name] = grad_W_i
          grads[affine_bias_name] = grad_b_i
          layer_idx -= 1
          # setting the parameters
          grads['W{}'.format(layer_number)]
          grads['b{}'.format(layer_number)]
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
