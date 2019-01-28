from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn

import networks

class Agent():

    hyperparams = {
        'agent_type': 'default',
        'input_size': 2,
        'num_classes': 3,
        'learning_rate': 0.00025,
        'optimizer': 'adamax',
        'seed': 123
    }

    def __init__(self, tensorflow_session, hyperparams):

        self.hyperparams.update(hyperparams)
        
        self.net_type = self.hyperparams['agent_type']          # Type of network to use
        self.n_input = self.hyperparams['input_size']           # Number of features in each element
        self.n_actions = self.hyperparams['num_classes']        # Number of output values
        self.learning_rate = self.hyperparams['learning_rate']  # Learning Rate
        self.optimiser_type = self.hyperparams['optimizer']     # Optimiser
        self.agent_seed = self.hyperparams['seed']              # Initial Seed

        # Tensorflow variables
        self.session = tensorflow_session
        self.state = tf.placeholder("float", [None, None, self.n_input])
        self.seq_len = tf.placeholder("int32", [None])
        self.masks = tf.placeholder("float", [None, None, 1])
        self.label = tf.placeholder("float", [None, self.n_actions])

        self.keep_prob = tf.Variable(0.5, trainable=False)

        # Get Network
        if self.net_type == 'default':
            embedding, _ = networks.set_network(self.state, self.masks,
                [[64, 64], [64,64]] )
            #embedding, _ = networks.test_network(self.state, self.masks)
            final, _ = networks.fc_network(embedding, [256, self.n_actions],
                keep_prob = self.keep_prob )
            self.pred = tf.nn.softmax(final)

        elif self.net_type == 'PCL':
            self.pred, self.weights = networks.PCL_network(self.state, self.masks,
                [self.n_input,256,256,256], [256,256,self.n_actions],
                keep_prob = self.keep_prob )
        
        elif self.net_type == 'pointnet':
            self.pred, self.weights = networks.point_network(self.state, self.masks,
                keep_prob = self.keep_prob )
                
        elif self.net_type == 'attention':
            embedding, _ = networks.attention_network(self.state, self.masks)
            final, _ = networks.fc_network(embedding, [256, self.n_actions],
                keep_prob = self.keep_prob )
            self.pred = tf.nn.softmax(final)

        elif self.net_type == 'RNN':
            self.pred, _ = networks.rnn_network(self.state, self.seq_len)

        elif self.net_type == 'simple':
            # Use full connected network
            state_ = tf.reshape(self.state, [-1, self.n_input*17])
            final, _ = networks.fc_network(state_, layer_sizes=[128, self.n_actions])
            self.pred = tf.nn.softmax(final)
            
        else:
            print "Invalid Network Type!"


        # Prediction accuracy
        correct_pred = tf.equal(tf.argmax(self.pred, 1),tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

        # Build Loss Function
        # Cross entropy for log-prob classification
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(tf.clip_by_value(self.pred,1e-10,1.0)), reduction_indices=[1]))

        # Optimiser
        loss = self.cross_entropy
        if self.optimiser_type == 'adam':
            optimiser = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optimiser_type == 'adamax':
            from adamax import AdamaxOptimizer
            optimiser = AdamaxOptimizer(self.learning_rate)
        elif self.optimiser_type == 'adamirror':
            from adamirror import AdamirrorOptimizer
            optimiser = AdamirrorOptimizer(self.learning_rate)
            print "Adamirror used"  
        else:
            print "Invalid Optimizer Type!"            

        self.compute_grads = optimiser.compute_gradients(loss)
        self.apply_grads = optimiser.apply_gradients(self.compute_grads)


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

        # Format Statistics
        statistics = { }

        return cross_entrophy, accuracy, statistics



# Turns the input list into a single tensor, padding the outputs.
# Produces a list of lengths of each list
# Produces a mask of size batch_size x max_list_len x 1
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






