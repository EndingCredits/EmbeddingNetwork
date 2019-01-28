from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn

from layers import * # sorry

################################################################################
############################## Standard networks ###############################
################################################################################

def feedforward(x, layers = [128], name="feedforward", **kwargs):
    """
    Concatenates a number of linear layers.
    """
    with tf.variable_scope(name):
        for i, layer in enumerate(layers):
            x = linear(x, layer, name='layer_' + str(i), **kwargs)

    return x


def statistic_network(x,
                      mask=None,
                      embedding_layers=[128]*2,
                      output_layers=[128]*2,
                      pool_type='max',
                      activation_fn=tf.nn.relu,
                      initializer=tf.truncated_normal_initializer(0, 0.02),
                      name="statistic_network"):
    """
    Simple statistic network which transforms inputs, pools them, and transforms
    result. Can also be used as a component in other networks.
    """

    # Rename for brevity
    act_fn = activation_fn

    if mask is None:
       mask = get_mask(x) # Get mask directly from state

    with tf.variable_scope(name):

        # Embedding Part:
        x = feedforward(x, embedding_layers, name="embedding", 
                        activation=act_fn, kernel_initializer=initializer)
        
        # Pool final elements to get input to task network
        x = pool(x, mask, pool_type=pool_type, keepdims=False)
        
        
        # Fully connected output (task) part:
        x = feedforward(x, output_layers, name="output", 
                        activation=act_fn, kernel_initializer=initializer)

    return x
    

def kary_network(x,
                 mask=None,
                 embedding_layers=[32]*2,
                 output_layers=[128]*2,
                 pool_type='max',
                 k=3,
                 activation_fn=tf.nn.relu,
                 initializer=tf.truncated_normal_initializer(0, 0.02),
                 name="kary_network"):
    """
    Simple statistic network which transforms inputs, pools them, and transforms
    result. Can also be used as a component in other networks.
    """

    # Rename for brevity
    act_fn = activation_fn

    if mask is None:
       mask = get_mask(x) # Get mask directly from state

    with tf.variable_scope(name):

        # Embedding Part:
        x = kary_pooling(x, k, embedding_layers, num_samples = 50, mask = mask,
          pool_type=pool_type,
                         name="kary_pooling", activation=act_fn,
                         initializer=initializer)
        
        # Fully connected output (task) part:
        x = feedforward(x, output_layers, name="output", 
                        activation=act_fn, kernel_initializer=initializer)

    return x

################################################################################
############################### Implementations ################################
################################################################################


def deep_sets_network(x,
                      mask=None,
                      embedding_layers = [256,256,256],
                      output_layers = [256],
                      use_equivariant=True,
                      initializer=tf.truncated_normal_initializer(0, 0.02),
                      activation_fn=tf.nn.relu,
                      name="deep_sets_network"):

    """
    The network of https://arxiv.org/abs/1611.04500. N.B, the parameters given
    here are different from those in the paper
    
    It consists of a number of linear element-wise transformations interspersed
    with 'submax' (i.e. f(x_i) = x_i - max_j{x_j}) equivariant transformations.
    
    This is then pooled to get a single vector representation of all objects and
    used as input for a final 'task' network.
    """
    
    if mask is None:
       mask = get_mask(x) # Get mask directly from state

    with tf.variable_scope(name):
        # Embedding Part:
        
        # Do the first layer without equivariant transform
        x = linear(x, embedding_layers[0], name='layer_' + str(0))
        
        # Do the rest of the embedding layers with 'submax' transform
        for i, layer in enumerate(embedding_layers[1:]):
            if use_equivariant:
                x = equiv_submax(x, mask)
            x = linear(x, layer, name='layer_' + str(i+1),
                activation=activation_fn, kernel_initializer=initializer )
        
        # Pool final elements to get input to task network
        x = pool(x, mask)
        
        # Fully connected (task) part:
        x = feedforward(x, output_layers, name="output", 
                       activation=activation_fn, kernel_initializer=initializer)
    
    # Returns the network output
    return x

def deep_sets_paper(x, mask=None, name="deep_sets_network"):
    # The parameters used in https://arxiv.org/abs/1611.04500.
    # N.B: No dropout used!
    return deep_sets_network(x, mask , name=name, activation_fn=tf.nn.tanh)


def pointnet(x, mask=None, name="pointnet"):
    """
    This replicates the full network of http://stanford.edu/~rqi/pointnet/
    """

    if mask is None:
       mask = get_mask(x) # Get mask directly from state

    def tnet(x, emb_layers, out_layers, name):
        with tf.variable_scope(name):
            context = statistic_network(x, mask, emb_layers, out_layers,
                                        activation_fn=tf.nn.relu)
            return transform_layer(x, context)

    # Build graph
    with tf.variable_scope(name):
        # Input T-Net
        x = tnet(x, [64, 128, 256], [256, 128], "input_tnet")
        # First block
        x = feedforward(x, [64, 64], name="block_0", activation=tf.nn.tanh)
        
        #Second T-Net
        x = tnet(x, [64, 128, 256], [256, 128], "tnet_1")
        # Second block
        x = feedforward(x, [64,256,256], name="block_1", activation=tf.nn.tanh)
            
        # Fully connected
        x = pool(x, mask)

        # Output part
        x = feedforward(x, [256, 256], name="output")
    
    return x


def object_embedding_network(x,
                             mask=None,
                             skip_style=False,
                             embedding_layers=[[128]*2]*2,
                             output_layers=[128]*3,
                             initializer=
                                  tf.truncated_normal_initializer(0, 0.02),
                             activation_fn=tf.nn.relu,
                             name="OENv0.5"):
    """
    The original embedding network used for object-based RL.
    
    Rather than using equivariant 'submax' layers this uses a 'context
    concatenation' approach, where the context (given by the max_pool) is
    concatenated with each element at the input to each block of layers.
    
    Rough diagram:
    
        x_i           Single element
      ___|___
     |       |
     |--128--|        1st 'block'
     |--128--|
     |_______|
      ___|___        
     |       |
     |    max_pool    
     |       |        Equivariant part
  f(x_i) o max_j x_j
      ___|___
     |       |
     |--128--|        2nd block
     |--128--|
     |_______|
         |
        ...
        
    This is then pooled to get a single vector representation of all objects and
    used as input for a final 'task' network.
    """
    
    if mask is None:
       mask = get_mask(x) # Get mask directly from state

    # Build graph:
    with tf.variable_scope(name):
    
        # Embedding Part:
        initial_elems = x
        for i, block in enumerate(embedding_layers):
          with tf.variable_scope("block_"+str(i)):
            if skip_style:
                # If skip-style we use the original elements for our input
                x = initial_elems
            for j, layer in enumerate(block):
                x = linear(x, layer, name='layer_' + str(j),
                           kernel_initializer=initializer)
                if j==0 and not i==0:
                    # If start of the next block we add in context
                    x = x + linear(c, layer, name='context',
                                   kernel_initializer=initializer)
                x = activation_fn(x)

            c = pool(x, mask, keepdims=True) # pool to get context


        # Fully connected (task) part:
        x = tf.squeeze(c, axis=-2)
        x = feedforward(x, output_layers, name="output", 
                        activation=act_fn, kernel_initializer=initializer)

    # Returns the network output
    return x
 


def relation_network(x,
                     mask=None,
                     num_blocks=2,
                     key_size=64,
                     value_size=128,
                     num_heads=4,
                     name="relation_network"):

    """
    A relational network using qkv self-attention
    
    Mimics the architecture given in Relational Deep Reinforcement Learning. 
    """


    # Keeping track of which initilizer and activation funcs are used
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu
    
    if mask is None:
       mask = get_mask(x) # Get mask directly from state
    
    # Embedding Part:

    # Add a number of attention blocks
    for i in range(num_blocks):
        x_att = attn_qkv(x, x, key_size, value_size, num_heads=num_heads,
                         mask=mask, name="attention_"+str(i), use_mlp_attn=False)
        #x_att = attn_mlp(x, x, key_size, value_size,
        #                  mask=mask, name="attention_"+str(i) )
        x_ff = feedforward(x, [value_size]*2, name="feedforward_"+str(i), 
                   activation=activation_fn, kernel_initializer=initializer)
        x = x_att + x_ff
    
    # Pool final elements to get input to task network
    x = pool(x, mask)

    # Fully connected (task) part:
    x = feedforward(x, [128]*2, name="output", 
                   activation=activation_fn, kernel_initializer=initializer)
                    

    # Returns the network output
    return x

def pseudo_relation_network(x,
                            mask=None,
                            num_clusters=10,
                            num_blocks=2,
                            key_size=64,
                            value_size=128,
                            num_heads=4,
                            name="pseudo_relation_network"):

    """
    A relational network using qkv self-attention
    
    Mimics the architecture given in Relational Deep Reinforcement Learning. 
    """


    # Keeping track of which initilizer and activation funcs are used
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu
    
    if mask is None:
       mask = get_mask(x) # Get mask directly from state
    
    # Embedding Part:

    # Add a number of attention blocks
    for i in range(num_blocks):
        clusters = tf.get_variable(name="clusters_"+str(i),
                                   shape=[num_clusters, key_size],
                                   initializer = tf.random_normal_initializer())

        clusters = tf.tile(tf.expand_dims(clusters,0), [shape_list(x)[0],1,1])


        c_att = attn_qkv(clusters, x, key_size, value_size, num_heads=num_heads,
                    mask=None, name="attention_A."+str(i), use_mlp_attn=False)

        x_att = attn_qkv(x, c_att, key_size, value_size, num_heads=num_heads,
                    mask=mask, name="attention_B."+str(i), use_mlp_attn=False)

        x_ff = feedforward(x, [value_size]*2, name="feedforward_"+str(i), 
                   activation=activation_fn, kernel_initializer=initializer)
        x = x_att + x_ff
    
    # Pool final elements to get input to task network
    x = pool(x, mask)

    # Fully connected (task) part:
    x = feedforward(x, [128]*2, name="output", 
                   activation=activation_fn, kernel_initializer=initializer)
                    

    # Returns the network output
    return x

################################################################################
############################### Legacy networks ################################
################################################################################


def __set_network(state, mask, layer_sizes=[[128,256]],
                  activation_function=tf.nn.relu, use_initial=False,
                  skip_connections=False, name='set_network'):
    params = []
    contexts = []
    
    # Embedding network
    initial_elems = state

    # Embedding network
    layer = initial_elems
    
    for i, block in enumerate(layer_sizes):
        for j, layer_size in enumerate(block):
            cont = context if j==0 and not i==0 else None
            layer, p = layers.invariant_layer(layer_size, layer, context=cont, name=name+'_l' + str(i) + '_'  + str(j))
            layer = activation_function(layer)
            params = params + p

        context = layers.mask_and_pool(layer, mask, 'MAX')
        #context = layers.att_pool(256, layer, mask, name=name+"_attn_"+str(i))

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
    
    
    
def __fc_network(state, layer_sizes = [256,256,10],
                 activation_function=tf.nn.relu, keep_prob=1.0, name='network'):
    params = []
    last_layer=len(layer_sizes)-1
    
    layer = state
    for i, layer_size in enumerate(layer_sizes):
        layer, p = layers.fc_layer(layer_size, layer, name=name+'_l_' + str(i))
        params = params + p
        if i!=last_layer:
            layer = activation_function(layer)
            layer = tf.nn.dropout(layer, keep_prob)
    
    # Output layer
    out = layer

    # Returns the network output and parameters
    return out, params



def __rnn_network(state, seq_len, d = [2,128,128,3]):
    num_layers = len(d)-2

    # Build graph
    lstm_cells = []
    for i in range(num_layers):
        lstm_cells.append(rnn.rnn_cell.GRUCell(d[i+1], activation=tf.nn.relu))
    multi_cell = rnn.rnn_cell.MultiRNNCell(lstm_cells)

    with tf.variable_scope("params") as vs:
      w = tf.Variable(tf.random_normal((d[-2],d[-1]), stddev=0.1), name='w_out')
      w_ = tf.Variable(tf.random_normal((d[-2],d[-1]), stddev=0.1), name='w_out_')
      b = tf.Variable(tf.zeros(d[-1]), name='b_out')
      output, _ = tf.nn.bidirectional_dynamic_rnn(multi_cell, multi_cell,
                      state, sequence_length = seq_len, dtype=tf.float32)

    last = layers.last_relevant(output[0], seq_len)
    first = layers.last_relevant(output[1], seq_len)
    prediction = tf.nn.softmax( tf.matmul(last, w) + tf.matmul(first, w_) + b )

    # Returns the network output, parameters, and the last layer as placeholder
    return prediction, _ #tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)
    
    

def __PCL_network(state, mask, emb_layer_sizes = [3,256,256,256],
                  net_layer_sizes = [256,40], keep_prob=0.5):
# This replicates the full network of https://arxiv.org/abs/1611.04500
    d = net_layer_sizes ; d_e = emb_layer_sizes
    num_layers = len(d)-1
    num_layers_e = len(d_e)-1

    # Set up params
    with tf.variable_scope("params") as vs:
        w_e = [None]*num_layers_e
        b_e = [None]*num_layers_e
        for i in range(num_layers_e):
            w_e[i] = tf.Variable(tf.random_normal((d_e[i],d_e[i+1]), stddev=0.1),
                         name='emb_c_w'+str(i+1))
            b_e[i] = tf.Variable(tf.zeros(d_e[i+1]), name='emb_b'+str(i+1))

        w_n = [None]*num_layers
        b_n = [None]*num_layers
        for i in range(num_layers):
            w_n[i] = tf.Variable(tf.random_normal((d[i],d[i+1]), stddev=0.1),
                         name='net_w'+str(i+1))
            b_n[i] = tf.Variable(tf.zeros(d[i+1]), name='net_b'+str(i+1))

    # Build graph

    # Embedding network
    elems = state
    for i in range(num_layers_e):
        pool = tf.matmul(layers.mask_and_pool(elems, mask), w_e[i])
        conv = tf.nn.conv1d(elems, [w_e[i]], stride=1, padding="SAME")
        elems = tf.nn.tanh(tf.reshape(pool,[-1,1,d_e[i+1]]) - conv + b_e[i])
    
    # Pool
    embed = layers.mask_and_pool(elems, mask)

    # Prediction network
    fc = tf.nn.dropout(embed, keep_prob)
    for i in range(num_layers-1):
        fc_ = tf.nn.tanh(tf.matmul(fc, w_n[i]) + b_n[i])
        fc = tf.nn.dropout(fc_, keep_prob)
    # Output layer
    predict = tf.nn.softmax( tf.matmul(fc, w_n[-1]) + b_n[-1] )

    # Regularisation for sparsity (average activation)
    rho = embed

    # Returns the network output, parameters
    return predict, w_e + b_e + w_n + b_n
    
    
    
def __point_network(state, mask, keep_prob=0.5):
# This replicates the full network of http://stanford.edu/~rqi/pointnet/

    batch_size = state.get_shape()[0]
    num_point = state.get_shape()[1]

    # Build graph

    elems = state
    
    # First Transform Net
    tranform = elems
    tranform, _ = set_network(tranform, mask, [[64, 128, 256]], name='tf_net_1a')
    tranform, _ = fc_network(tranform, [256, 128, 3*3], name='tf_net_1b')
    tranform = tf.reshape(tranform, [-1, 3, 3])
    tranform += tf.constant(np.eye(3), dtype=tf.float32)
    elems = tf.matmul(elems, tranform)
    
    # First block
    elems, _ = layers.invariant_layer(64, elems, name='block_1a')
    elems = tf.nn.tanh(elems)
    elems, _ = layers.invariant_layer(64, elems, name='block_1b')
    elems = tf.nn.tanh(elems)
    
    # Second Transform Net
    tranform = elems
    tranform, _ = set_network(tranform, mask, [[64, 128, 256]], name='tf_net_2a')
    tranform, _ = fc_network(tranform, [256, 128, 64*64], name='tf_net_2b')
    tranform = tf.reshape(tranform, [-1, 64, 64])
    tranform += tf.constant(np.eye(64), dtype=tf.float32)
    elems = tf.matmul(elems, tranform)
    
    # Second block
    elems, _ = layers.invariant_layer(64, elems, name='block_2a')
    elems = tf.nn.tanh(elems)
    elems, _ = layers.invariant_layer(256, elems, name='block_2b')
    elems = tf.nn.tanh(elems)
    elems, _ = layers.invariant_layer(256, elems, name='block_3c')
    elems = tf.nn.tanh(elems)
        
    # Fully connected
    embed = layers.mask_and_pool(elems, mask)
    final, _ = fc_network(embed, [256, 256, 40], name='out', keep_prob=keep_prob)
    
    predict = tf.nn.softmax(final)

    # Returns the network output, parameters
    return predict, []


