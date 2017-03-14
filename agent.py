from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn


class Agent():

    num_agents = 0
    default_params = {
        'agent_type': 'embed',
        'input_size': 2,
        'num_classes': 3,
        'embedding_size': 256,
        'learning_rate': 0.0025,
        'rho_target': 0.05,
        'sparsity_reg': 0.0,
        'seed': 123
    }

    def __init__(self, tensorflow_session, hyperparams):

        self.hyperparams = hyperparams
        self.net_type = hyperparams['agent_type']          # Type of network to use
        self.n_input = hyperparams['input_size']           # Number of features in each element
        self.n_actions = hyperparams['num_classes']        # Number of output values
        self.e_layer_size = hyperparams['embedding_size']  # Size of embedding layer
        self.learning_rate = hyperparams['learning_rate']  # Learning Rate
        self.rho_target = hyperparams['rho_target']        # Target average activation
        self.sparsity_reg = hyperparams['sparsity_reg']    # Strength of sparsity regularisation
        
        self.seed = hyperparams['seed']                    # Seed
        self.curr_seed = self.seed
        self.agent_num = Agent.num_agents ; Agent.num_agents += 1

        # Tensorflow variables
        self.session = tensorflow_session
        self.state = tf.placeholder("float", [None, None, self.n_input])
        self.label = tf.placeholder("float", [None, self.n_actions])
        self.seq_len = tf.placeholder("int32", [None])
        self.masks = tf.placeholder("float", [None, None, 1])

        self.keep_prob = tf.Variable(0.5, trainable=False)

        # Get Network
        if self.net_type == 'reembedding':
            self.pred, self.weights = self.embedding_network(self.state, self.masks,
                #[[self.n_input,256,self.e_layer_size]], [self.e_layer_size,128,self.n_actions])
                [[self.n_input,256],[self.n_input,256],[self.n_input,256],[self.n_input,self.e_layer_size]], [self.e_layer_size,256,self.n_actions],
                keep_prob = self.keep_prob )

        elif self.net_type == 'reembedding_full':
            self.pred, self.weights = self.embedding_network(self.state, self.masks,
                [[self.n_input,256],[256,256],[256,256],[256,self.e_layer_size]], [self.e_layer_size,256,self.n_actions],
                use_initial = False, keep_prob = self.keep_prob )

        elif self.net_type == 'PCL':
            self.pred, self.weights = self.PCL_network(self.state, self.masks,
                [self.n_input,256,256,256], [256,256,self.n_actions],
                keep_prob = self.keep_prob )

        elif self.net_type == 'RNN':
            self.pred, self.weights = self.rnn(self.state, self.seq_len)

        elif self.net_type == 'simple':
            self.pred, self.weights = self.fc_network(self.state)

        else:
            self.pred, self.weights = self.embedding_network(self.state, self.masks,
                [[self.n_input,128,self.e_layer_size]], [self.e_layer_size,128,self.n_actions])

        # Prediction accuracy
        correct_pred = tf.equal(tf.argmax(self.pred, 1),tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

        # Build Loss Function

        # Cross entropy for log-prob classification
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(tf.clip_by_value(self.pred,1e-10,1.0)), reduction_indices=[1]))

        # Sparsity Regularisation
        #self.rho_mean = tf.reduce_mean(self.rho)
        #self.KL_p = self.rho_target * (tf.log(self.rho_target) - tf.log(tf.clip_by_value(self.rho,1e-10,1.0))) \
        #             + (1-self.rho_target) * (tf.log(1-self.rho_target) - tf.log(tf.clip_by_value(1-self.rho,1e-10,1.0)))
        #self.KL = tf.reduce_sum(self.KL_p)
        #self.l2 = tf.sqrt(tf.reduce_sum(self.rho*self.rho))
        #self.l1 = tf.reduce_sum(tf.abs(self.rho))
        #self.pq_mean = self.l1 / self.l2
        #self.pq_mean_sq = tf.reduce_sum(self.rho) * tf.reduce_sum(self.rho) / tf.reduce_sum(self.rho*self.rho) / tf.to_float(tf.size(self.rho))

        # Optimiser
        loss = self.cross_entropy #+ self.sparsity_reg*self.pq_mean_sq
        optimiser = tf.train.AdamOptimizer(self.learning_rate)
        self.compute_grads = optimiser.compute_gradients(loss)
        self.apply_grads = optimiser.apply_gradients(self.compute_grads)

    def get_seed(self):
        self.curr_seed += 1
        return self.curr_seed

    def predict(self, state):
        # Repackage State using util function
        state_, l, m = batchToArrays(state)

        # Get prediction and other statistics
        pred = self.session.run([self.pred], feed_dict={self.state: state_, self.seq_len: l, self.masks: m})

        statistics = { }

        return pred, statistics


    def test(self, state, label):

        # Repackage State using util function
        state_, l, m = batchToArrays(state)

        # Get and apply grads, cross entrophy (i.e. pediction loss) and prediction accuracy, as well as other stats
        cross_entrophy, accuracy = self.session.run([self.cross_entropy, self.accuracy],
            feed_dict={self.state: state_, self.seq_len: l, self.masks: m, self.label: label, self.keep_prob: 1.0})

        # Format Statistics
        statistics = { }

        return cross_entrophy, accuracy, statistics


    def train(self, state, label):

        # Repackage State using util function
        state_, l, m = batchToArrays(state)

        # Get and apply grads, cross entrophy (i.e. pediction loss) and prediction accuracy, as well as other stats
        gvs, cross_entrophy, accuracy = self.session.run([self.apply_grads, self.cross_entropy, self.accuracy],
            feed_dict={self.state: state_, self.seq_len: l, self.masks: m, self.label: label})

        # Update grads, with janky grads code...
        #feed_dict_ = {}
        #for i, grad_var in enumerate(gvs):
        #    feed_dict_[self.compute_grads[i][0]] = grad_var[0]
        #self.session.run(self.apply_grads, feed_dict=feed_dict_)

        # Format Statistics
        statistics = { }

        return cross_entrophy, accuracy, statistics



    def embedding_network(self, state, mask, emb_layer_sizes = [[2,128,256]], net_layer_sizes = [256,128,3], use_initial=True, keep_prob=1.0):
    # This could probably be moved into a 'models' file
        d = net_layer_sizes
        d_e = emb_layer_sizes
        num_layers = len(d)-1
        num_e_layers = len(d_e)

        # Set up params
        with tf.variable_scope("params_agent"+str(self.agent_num)) as vs:
            #Embedding parts
            w_e = [] ; b_e = []
            for i in range(num_e_layers):
                layer_w = [] ; layer_b = []
                layer = d_e[i]
                
                for j in range(len(layer)-1):
                    layer_w.append( tf.Variable(tf.random_normal((layer[j],layer[j+1]), stddev=0.1), name='emb_w'+str(i)+'_'+str(j+1)) )
                    layer_b.append( tf.Variable(tf.zeros(layer[j+1]), name='emb_b'+str(i)+'_'+str(j+1)) )
                w_e.append( layer_w ) ; b_e.append( layer_b )

            # Combining
            w_e_c = []
            for i in range(num_e_layers-1):
                w_e_c.append( tf.Variable(tf.random_normal((d_e[i][-1],d_e[i+1][1]), stddev=0.1), name='emb_w_c'+str(i)) )

            # Final part
            w_n = [None]*num_layers
            b_n = [None]*num_layers
            for i in range(num_layers):
                w_n[i] = tf.Variable(tf.random_normal((d[i],d[i+1]), stddev=0.1, seed=self.get_seed()), name='net_w'+str(i+1))
                b_n[i] = tf.Variable(tf.zeros(d[i+1]), name='net_b'+str(i+1))


        # Build graph:

        # Embedding network

        initial_elems = state

        ### Legacy Method
        #  w_1 = tf.Variable(tf.random_normal((2,64), stddev=0.1))
        #  b_1 = tf.Variable(tf.zeros((64)))
        #  w_2 = tf.Variable(tf.random_normal((64,EMBEDDING_LAYER_SIZE), stddev=0.1))
        #  b_2 = tf.Variable(tf.zeros((EMBEDDING_LAYER_SIZE)))
        # Need to transpose so that embedding applies along objects in batch
        #  elems = tf.transpose(state, [1, 0, 2])
        #  embeds = tf.map_fn(lambda x: tf.nn.relu(tf. matmul(tf.nn.relu(tf.matmul(x, w_1) + b_1), w_2)+b_2), elems, parallel_iterations=8)
        #  embeds = tf.mul(embeds, tf.transpose(mask, [1, 0, 2]))
        #  embed = tf.reduce_max(embeds, 0)

        #Initial Embedding
        elems = initial_elems
        w = w_e[0] ; b = b_e[0]
        for i in range(len(w)):
            elems = tf.nn.relu(tf.nn.conv1d(elems, [w[i]], stride=1, padding="SAME") + b[i])

        #Rembeddings
        for i in range(num_e_layers-1):
            w = w_e[i+1] ; b = b_e[i+1] ; w_c = w_e_c[i]
            # Initial layer
            pool = tf.matmul(mask_and_pool(elems, mask), w_c)
            starting_elems = initial_elems if use_initial else elems
            conv = tf.nn.conv1d(starting_elems, [w[0]], stride=1, padding="SAME")
            elems = tf.nn.relu(tf.reshape(pool,[-1,1,d_e[i+1][1]]) + conv + b[0])
            # Other layers
            for j in range(len(w)-1):
                elems = tf.nn.relu(tf.nn.conv1d(elems, [w[j+1]], stride=1, padding="SAME") + b[j+1])

        embed = mask_and_pool(elems, mask)

        # Prediction network

        fc = tf.nn.dropout(embed, keep_prob)
        for i in range(num_layers-1):
            fc_ = tf.nn.relu(tf.matmul(fc, w_n[i]) + b_n[i])
            fc = tf.nn.dropout(fc_, keep_prob)

        # Output layer
        predict = tf.nn.softmax( tf.matmul(fc, w_n[-1]) + b_n[-1] )

        # Returns the network output, parameters, object embeddings, and representation layer
        return predict, tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)



    def PCL_network(self, state, mask, emb_layer_sizes = [3,256,256,256], net_layer_sizes = [256,40,3], keep_prob=0.5):
    # This could probably be moved into a 'models' file
    # This replicates the network of https://arxiv.org/abs/1611.04500
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



    def rnn(self, state, seq_len, d = [2,128,128,3]):
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

        last = last_relevant(output[0], seq_len)
        first = last_relevant(output[1], seq_len)
        prediction = tf.nn.softmax( tf.matmul(last, w) + tf.matmul(first, w_) + b )

        # Returns the network output, parameters, and the last layer as placeholder
        return prediction, tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)



    def fc_network(self, state, d = [2*20,256,256,3]):
        num_layers = len(d)-1

        w = [None]*num_layers
        b = [None]*num_layers
        for i in range(num_layers):
            w[i] = tf.Variable(tf.random_normal((d[i],d[i+1]), stddev=0.1, seed=self.get_seed()), name='w'+str(i+1))
            b[i] = tf.Variable(tf.zeros(d[i+1]), name='b'+str(i+1))

        # Prediction network
        fc = tf.reshape(state, [-1, d[0]])
        for i in range(num_layers-1):
            fc = tf.nn.relu(tf.matmul(fc, w[i]) + b[i])
        
        # Output layer
        prediction = tf.nn.softmax( tf.matmul(fc, w[-1]) + b[-1] )

        # Returns the network output, parameters, and the last layer as placeholder
        return prediction, w + b


def batchToArrays(input_list):
    max_len = 0
    out = []; seq_len = []; masks = []
    for i in input_list: max_len = max(len(i),max_len)
    for l in input_list:
        # Zero pad output
        out.append(np.pad(np.array(l,dtype=np.float32), ((0,max_len-len(l)),(0,0)), mode='constant'))
        seq_len.append(len(l))
        # Create mask...
        masks.append(np.pad(np.array(np.ones((len(l),1)),dtype=np.float32), ((0,max_len-len(l)),(0,0)), mode='constant'))
    return out, seq_len, masks


def mask_and_pool(embeds, mask):
    # Use broadcasting to multiply
    masked_embeds = tf.multiply(embeds, mask)

    # Pool using max pooling
    embed = tf.reduce_max(masked_embeds, 1)

    # For mean pooling:
    #embed = tf.reduce_sum(masked_embeds, 1) / tf.reduce_sum(mask, 1)

    return embed



def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

#Legacy code
def mask_and_pool_(embeds, seq_len):
    # Masking code shamelessly ripped off stack overflow...
    # This is actually pretty slow, there's probably a better way to do this...

    # Make a matrix where each row contains the length
    max_len = tf.reduce_max(seq_len)
    lengths_transposed = tf.expand_dims(seq_len, 1)

    # Make a matrix where each row contains [0, 1, ..., l]
    range_ = tf.range(0, max_len, 1)
    range_row = tf.expand_dims(range_, 0)

    # Use the logical operations to create a mask
    mask = tf.to_float(tf.less(range_row, lengths_transposed))
    
    # Use broadcasting to multiply
    masked_embeds = tf.multiply(embeds, tf.expand_dims(mask,2))

    # Pool using max pooling
    embed = tf.reduce_max(masked_embeds, 1)

    # For mean pooling:
    #embed = tf.reduce_sum(masked_embeds, 1) / tf.reduce_sum(mask, 1)

    return embed


