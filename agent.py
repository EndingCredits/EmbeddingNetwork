from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


class Agent():

    num_agents = 0

    def __init__(self, tensorflow_session, hyperparams):

        self.hyperparams = hyperparams
        self.net_type = hyperparams['agent_type']          # Type of network to use
        self.n_input = hyperparams['input_size']           # Number of features in each element
        self.n_actions = hyperparams['num_classes']        # Number of output values
        self.e_layer_size = hyperparams['embedding_size']  # Size of embedding layer
        self.learning_rate = hyperparams['learning_rate']  # Learning Rate
        self.rho_target = hyperparams['rho_target']        # Target average activation
        self.sparsity_reg = hyperparams['sparsity_reg']    # Strength of sparsity regularisation

        self.agent_num = Agent.num_agents ; Agent.num_agents += 1

        # Tensorflow variables
        self.session = tensorflow_session
        self.state = tf.placeholder("float", [None, None, self.n_input])
        self.label = tf.placeholder("float", [None, self.n_actions])
        self.seq_len = tf.placeholder("int32", [None])
        self.masks = tf.placeholder("float", [None, None, self.e_layer_size])

        # Get Network
        if self.net_type == 'RNN':
            self.pred, self.weights, self.rho, self.embed = self.rnn(self.state, self.seq_len)
        elif self.net_type == 'reembedding':
            self.pred, self.weights, self.rho, self.embed = self.reembedding_network(self.state, self.masks, [2,128,self.e_layer_size], [self.e_layer_size,128,3])
        elif self.net_type == 'simple':
            self.pred, self.weights, self.rho, self.embed = self.fc_network(self.state)
        else:
            self.pred, self.weights, self.rho, self.embed = self.network(self.state, self.masks, [2,128,self.e_layer_size], [self.e_layer_size,128,3])

        # Prediction accuracy
        correct_pred = tf.equal(tf.argmax(self.pred, 1),tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

        # Build Loss Function

        # Cross entropy for log-prob classification
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(tf.clip_by_value(self.pred,1e-10,1.0)), reduction_indices=[1]))

        # Sparsity Regularisation
        self.rho_mean = tf.reduce_mean(self.rho)
        self.KL_p = self.rho_target * (tf.log(self.rho_target) - tf.log(tf.clip_by_value(self.rho,1e-10,1.0))) \
                     + (1-self.rho_target) * (tf.log(1-self.rho_target) - tf.log(tf.clip_by_value(1-self.rho,1e-10,1.0)))
        self.KL = tf.reduce_sum(self.KL_p)
        self.l2 = tf.sqrt(tf.reduce_sum(self.rho*self.rho))
        self.l1 = tf.reduce_sum(tf.abs(self.rho))
        self.pq_mean = self.l1 / self.l2

        # Optimiser
        loss = self.cross_entropy + self.sparsity_reg*self.pq_mean
        optimiser = tf.train.AdamOptimizer(self.learning_rate)
        self.compute_grads = optimiser.compute_gradients(loss)
        self.apply_grads = optimiser.apply_gradients(self.compute_grads)


    def predict(self, state):
        # Repackage State using util function
        state_, l, m = batchToArrays(state, self.e_layer_size)

        # Get prediction and other statistics
        pred, rho, embed = self.session.run([self.pred, self.rho_mean, self.embed], feed_dict={self.state: state_, self.seq_len: l, self.masks: m})

        statistics = { 'rho_mean': rho, 'embedding': embed }

        return pred, statistics


    def train(self, state, label):

        # Repackage State using util function
        state_, l, m = batchToArrays(state, self.e_layer_size)

        # Get and apply grads, cross entrophy (i.e. pediction loss) and prediction accuracy, as well as other stats
        gvs, cross_entrophy, accuracy, rho, embed = self.session.run([self.apply_grads, self.cross_entropy, self.accuracy, self.rho_mean, self.embed],
            feed_dict={self.state: state_, self.seq_len: l, self.masks: m, self.label: label})

        # Update grads, with janky grads code...
        #feed_dict_ = {}
        #for i, grad_var in enumerate(gvs):
        #    feed_dict_[self.compute_grads[i][0]] = grad_var[0]
        #self.session.run(self.apply_grads, feed_dict=feed_dict_)
        

        # Format Statistics
        statistics = { 'rho_mean': rho, 'embedding': embed }

        return cross_entrophy, accuracy, statistics


    def get_embedding(self, state):
        # Repackage State using util function
        state_, l, m = batchToArrays(state, self.e_layer_size)

        # Get embedding
        embed = self.session.run(self.embed, feed_dict={self.state: state_, self.seq_len: l, self.masks: m})

        return embed


    def network(self, state, mask, emb_layer_sizes = [2,64,256], net_layer_sizes = [256,64,3]):
    # This could probably be moved into a 'models' file
        d = net_layer_sizes ; d_e = emb_layer_sizes
        num_layers = len(d)-1
        num_layers_e = len(d_e)-1

        # Set up params
        with tf.variable_scope("params_agent"+str(self.agent_num)) as vs:

            w_e = [None]*num_layers_e
            b_e = [None]*num_layers_e
            for i in range(num_layers_e):
                w_e[i] = tf.Variable(tf.random_normal((d_e[i],d_e[i+1]), stddev=0.1), name='emb_w'+str(i+1))
                b_e[i] = tf.Variable(tf.zeros(d_e[i+1]), name='emb_b'+str(i+1))

            w_n = [None]*num_layers
            b_n = [None]*num_layers
            for i in range(num_layers):
                w_n[i] = tf.Variable(tf.random_normal((d[i],d[i+1]), stddev=0.1), name='net_w'+str(i+1))
                b_n[i] = tf.Variable(tf.zeros(d[i+1]), name='net_b'+str(i+1))

        # Build graph

        # Embedding network
        conv = state
        for i in range(num_layers_e):
            conv = tf.nn.relu(tf.nn.conv1d(conv, [w_e[i]], stride=1, padding="SAME") + b_e[i])
        embeds = conv#tf.nn.softmax(tf.nn.conv1d(conv, [w_e[-1]], stride=1, padding="SAME") + b_e[-1])#conv

        # Mask embeds so that we remove anything from 0-padded inputs
        embeds = tf.mul(embeds, mask)

        # Pool along objects dimension
        # Using nn pooling is slightly faster than tf.reduce_max
        embed_ = tf.nn.max_pool([embeds], ksize=[1, 1, 1000, 1], strides=[1, 1, 1000, 1], padding="SAME")
        embed = tf.reshape(embed_, [-1, self.e_layer_size])
        #embed = tf.reduce_sum(embeds, 1) / tf.reduce_sum(mask, 1)

        # Prediction network
        fc = embed
        for i in range(num_layers-1):
            fc = tf.nn.relu(tf.matmul(fc, w_n[i]) + b_n[i])
        # Output layer
        predict = tf.nn.softmax( tf.matmul(fc, w_n[-1]) + b_n[-1] )

        # Regularisation for sparsity (average activation)
        #rho = tf.reduce_mean(embed, 1)
        rho = tf.reduce_sum(embeds, [0, 1]) / tf.reduce_sum(mask, [0, 1])

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

        # Returns the network output, parameters, object embeddings, and representation layer
        return predict, w_e + b_e + w_n + b_n, rho, embed


    def reembedding_network(self, state, mask, emb_layer_sizes = [2,128,256], net_layer_sizes = [256,128,3]):
    # This could probably be moved into a 'models' file
        d = net_layer_sizes ; d_e = emb_layer_sizes
        num_layers = len(d)-1
        num_layers_e = len(d_e)-1

        # Set up params
        with tf.variable_scope("params_agent"+str(self.agent_num)) as vs:

            w_e = [None]*num_layers_e
            b_e = [None]*num_layers_e
            for i in range(num_layers_e):
                w_e[i] = tf.Variable(tf.random_normal((d_e[i],d_e[i+1]), stddev=0.1), name='emb_w'+str(i+1))
                b_e[i] = tf.Variable(tf.zeros(d_e[i+1]), name='emb_b'+str(i+1))

            w_n = [None]*num_layers
            b_n = [None]*num_layers
            for i in range(num_layers):
                w_n[i] = tf.Variable(tf.random_normal((d[i],d[i+1]), stddev=0.1), name='net_w'+str(i+1))
                b_n[i] = tf.Variable(tf.zeros(d[i+1]), name='net_b'+str(i+1))
            w_n_e = tf.Variable(tf.random_normal((d[0],d[1]), stddev=0.1), name='net_w_e')
            #w_n.append(w_n_e)

            w_re = [] ; b_re = []
            w_re_1 = tf.Variable(tf.random_normal((2,64), stddev=0.1), name='reemb_w_1')
            w_re_e = tf.Variable(tf.random_normal((self.e_layer_size,64), stddev=0.1), name='reemb_w_e')
            b_re_1 = tf.Variable(tf.zeros(64), name='reemb_b_1')
            w_re_2 = tf.Variable(tf.random_normal((64,self.e_layer_size), stddev=0.1), name='reemb_w_2')
            b_re_2 = tf.Variable(tf.zeros(self.e_layer_size), name='reemb_b_2')
            w_re.append(w_re_1) ; w_re.append(w_re_e) ; w_re.append(w_re_2)
            b_re.append(b_re_1) ; b_re.append(b_re_2) 

        # Build graph

        # Embedding network
        conv = state
        for i in range(num_layers_e):
            conv = tf.nn.relu(tf.nn.conv1d(conv, [w_e[i]], stride=1, padding="SAME") + b_e[i])
        #conv = tf.nn.softmax(tf.nn.conv1d(conv, [w_e[-1]], stride=1, padding="SAME") + b_e[-1])
        init_embeds = tf.mul(conv, mask)
        init_embed_ = tf.nn.max_pool([init_embeds], ksize=[1, 1, 1000, 1], strides=[1, 1, 1000, 1], padding="SAME")
        init_embed = tf.reshape(init_embed_, [-1, self.e_layer_size])

        # Rembedding network
        elements_part = tf.nn.conv1d(state, [w_re_1], stride=1, padding="SAME")
        embeddings_part = tf.reshape(tf.matmul(init_embed, w_re_e), [-1,1,64])
        h_layer = tf.nn.relu( elements_part + embeddings_part + b_re_1 , name='hidden_layer')
        o_layer = tf.nn.relu( tf.nn.conv1d(h_layer, [w_re_2], stride=1, padding="SAME") + b_re_2 )
        embeds = tf.mul(o_layer, mask)
        embed_ = tf.nn.max_pool([embeds], ksize=[1, 1, 1000, 1], strides=[1, 1, 1000, 1], padding="SAME")
        embed = tf.reshape(embed_, [-1, self.e_layer_size])

        # Prediction network
        fc = tf.nn.relu(tf.matmul(embed, w_n[0]) + tf.matmul(init_embed, w_n_e) + b_n[0])
        for i in range(num_layers-2):
            fc = tf.nn.relu(tf.matmul(fc, w_n[i+1]) + b_n[i+1])
        # Output layer
        predict = tf.nn.softmax( tf.matmul(fc, w_n[-1]) + b_n[-1] )

        # Regularisation for sparsity (average activation)
        rho = tf.reduce_sum(embeds, [0, 1]) / tf.reduce_sum(mask, [0, 1])

        # Returns the network output, parameters, object embeddings, and representation layer
        return predict, w_e + b_e + w_n + b_n, rho, embed


    def rnn(self, state, seq_len, d = [2,128,128,3]):
        num_layers = len(d)-2

        # Build graph
        lstm_cells = []
        for i in range(num_layers): lstm_cells.append(rnn_cell.LSTMCell(d[i+1]))
        multi_cell = rnn_cell.MultiRNNCell(lstm_cells)

        with tf.variable_scope("params_agent"+str(self.agent_num)) as vs:
          w = tf.Variable(tf.random_normal((d[-2],d[-1]), stddev=0.1), name='w_out')
          b = tf.Variable(tf.zeros(d[-1]), name='b_out')
          output, _ = tf.nn.dynamic_rnn(multi_cell, state, sequence_length = seq_len, dtype=tf.float32)

        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, tf.shape(output)[0] - 1)
        prediction = tf.nn.softmax( tf.matmul(last, w) + b )

        # Returns the network output, parameters, and the last layer as placeholder
        return prediction, tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name), last, last


    def fc_network(self, state, d = [2*20,256,256,3]):
        num_layers = len(d)-1

        w = [None]*num_layers
        b = [None]*num_layers
        for i in range(num_layers):
            w[i] = tf.Variable(tf.random_normal((d[i],d[i+1]), stddev=0.1), name='w'+str(i+1))
            b[i] = tf.Variable(tf.zeros(d[i+1]), name='b'+str(i+1))

        # Prediction network
        fc = tf.reshape(state, [-1, d[0]])
        for i in range(num_layers-1):
            fc = tf.nn.relu(tf.matmul(fc, w[i]) + b[i])
        
        # Output layer
        prediction = tf.nn.softmax( tf.matmul(fc, w[-1]) + b[-1] )

        # Returns the network output, parameters, and the last layer as placeholder
        return prediction, w + b, fc, fc


def batchToArrays(input_list, mask_size):
    max_len = 0
    out = []; seq_len = []; masks = []
    for i in input_list: max_len = max(len(i),max_len)
    for l in input_list:
        out.append(np.pad(np.array(l,dtype=np.float32), ((0,max_len-len(l)),(0,0)), mode='constant'))
        seq_len.append(len(l))
        masks.append(np.pad(np.array(np.ones((len(l),mask_size)),dtype=np.float32), ((0,max_len-len(l)),(0,0)), mode='constant'))
    return out, seq_len, masks



