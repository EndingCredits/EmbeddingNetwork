from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn

import layers

def set_network(state, mask, layer_sizes=[[128,256]], activation_function=tf.nn.relu, use_initial=False, skip_connections=False):
    params = []
    contexts = []
    
    # Embedding network
    initial_elems = state

    # Embedding network
    layer = initial_elems

    for i, block in enumerate(layer_sizes):
        for j, layer_size in enumerate(block):
            cont = context if j==0 and not i==0 else None
            layer, p = layers.invariant_layer(layer_size, layer, context=cont, name='l' + str(i) + '_'  + str(j))
            layer = activation_function(layer)
            params = params + p

        context = layers.mask_and_pool(layer, mask)

        if skip_connections:
            contexts.append(context)

        # Reset current inputs to original elements if required
        if use_initial:
            layer = initial_elems
    
    if skip_connections:
        # Concatenate all contexts
        out = tf.concat(contexts, axis=1)
    else:
        # Just use the last one
        out = context

    # Returns the network output and parameters
    return out, params
    
    
def fc_network(state, layer_sizes = [256,256,10], activation_function=tf.nn.relu, keep_prob=1.0):
    params = []
    last_layer=len(layer_sizes)-1
    
    layer = state
    for i, layer_size in enumerate(layer_sizes):
        layer, p = layers.fc_layer(layer_size, layer, name='l_' + str(i))
        params = params + p
        if i!=last_layer:
            layer = activation_function(layer)
            layer = tf.nn.dropout(layer, keep_prob)
    
    # Output layer
    out = layer

    # Returns the network output and parameters
    return out, params


def rnn_network(state, seq_len, d = [2,128,128,3]):
    num_layers = len(d)-2

    # Build graph
    lstm_cells = []
    for i in range(num_layers): lstm_cells.append(rnn.rnn_cell.GRUCell(d[i+1], activation=tf.nn.relu))
    multi_cell = rnn.rnn_cell.MultiRNNCell(lstm_cells)

    with tf.variable_scope("params_agent"+str(self.agent_num)) as vs:
      w = tf.Variable(tf.random_normal((d[-2],d[-1]), stddev=0.1), name='w_out')
      w_ = tf.Variable(tf.random_normal((d[-2],d[-1]), stddev=0.1), name='w_out_')
      b = tf.Variable(tf.zeros(d[-1]), name='b_out')
      output, _ = tf.nn.bidirectional_dynamic_rnn(multi_cell, multi_cell, state, sequence_length = seq_len, dtype=tf.float32)

    last = layers.last_relevant(output[0], seq_len)
    first = layers.last_relevant(output[1], seq_len)
    prediction = tf.nn.softmax( tf.matmul(last, w) + tf.matmul(first, w_) + b )

    # Returns the network output, parameters, and the last layer as placeholder
    return prediction, _ #tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)
    

def PCL_network(state, mask, emb_layer_sizes = [3,256,256,256], net_layer_sizes = [256,40,3], keep_prob=0.5):
# This replicates the full network of https://arxiv.org/abs/1611.04500
    d = net_layer_sizes ; d_e = emb_layer_sizes
    num_layers = len(d)-1
    num_layers_e = len(d_e)-1

    # Set up params
    with tf.variable_scope("params_agent"+str(self.agent_num)) as vs:
        w_e = [None]*num_layers_e
        b_e = [None]*num_layers_e
        for i in range(num_layers_e):
            w_e[i] = tf.Variable(tf.random_normal((d_e[i],d_e[i+1]), stddev=0.1, seed=self.get_seed()), name='emb_c_w'+str(i+1))
            b_e[i] = tf.Variable(tf.zeros(d_e[i+1]), name='emb_b'+str(i+1))

        w_n = [None]*num_layers
        b_n = [None]*num_layers
        for i in range(num_layers):
            w_n[i] = tf.Variable(tf.random_normal((d[i],d[i+1]), stddev=0.1, seed=self.get_seed()), name='net_w'+str(i+1))
            b_n[i] = tf.Variable(tf.zeros(d[i+1]), name='net_b'+str(i+1))

    # Build graph

    # Embedding network
    elems = state
    for i in range(num_layers_e):
        pool = tf.matmul(mask_and_pool(elems, mask), w_e[i])
        conv = tf.nn.conv1d(elems, [w_e[i]], stride=1, padding="SAME")
        elems = tf.nn.tanh(tf.reshape(pool,[-1,1,d_e[i+1]]) - conv + b_e[i])
    
    # Pool
    embed = mask_and_pool(elems, mask)

    # Prediction network
    fc = tf.nn.dropout(embed, keep_prob)
    for i in range(num_layers-1):
        fc_ = tf.nn.tanh(tf.matmul(fc, w_n[i]) + b_n[i])
        fc = tf.nn.dropout(fc_, keep_prob)
    # Output layer
    predict = tf.nn.softmax( tf.matmul(fc, w_n[-1]) + b_n[-1] )

    # Regularisation for sparsity (average activation)
    rho = embed

    # Returns the network output, parameters, object embeddings, and representation layer
    return predict, w_e + b_e + w_n + b_n






