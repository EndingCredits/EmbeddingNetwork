from __future__ import division

import argparse
import os
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

import matplotlib.pyplot as plt

EMBEDDING_LAYER_SIZE = 256

class Agent():
    def __init__(self, session, args):

        self.n_input = args.input_size     # Number of features in each element
        self.n_actions = args.num_actions  # Number of output values
        self.learning_rate = args.learning_rate
        self.rho_target = 0.05


        # Tensorflow variables
        self.session = session
        self.state = tf.placeholder("float", [None, None, self.n_input])
        self.label = tf.placeholder("float", [None, self.n_actions])
        self.seq_len = tf.placeholder("int32", [None])
        self.masks = tf.placeholder("float", [None, None, EMBEDDING_LAYER_SIZE])

        # Get Network
        self.pred, self.weights, self.rho, self.embed = self.network(self.state, self.masks)
        #self.pred, self.weights, self.rho, self.embed = self.rnn(self.state, self.seq_len)

        # Build Loss Function

        # Sparsity Regularisation
        self.rho_mean = tf.reduce_mean(self.rho)
        self.embL1 = tf.reduce_sum(tf.abs(self.rho))
        self.KL_p = self.rho_target * (tf.log(self.rho_target) - tf.log(tf.clip_by_value(self.rho,1e-10,1.0))) \
                     + (1-self.rho_target) * (tf.log(1-self.rho_target) - tf.log(tf.clip_by_value(1-self.rho,1e-10,1.0)))
        self.KL = tf.reduce_sum(self.KL_p)

        # Cross entropy for log-prob classification
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(tf.clip_by_value(self.pred,1e-10,1.0)), reduction_indices=[1]))

        # Optimiser
        loss = self.cross_entropy #+ 0.1*self.KL
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)


    def predict(self, state):
        state_, l, m = batchToArrays(state)
        pred, embed = self.session.run([self.pred, self.embed], feed_dict={self.state: state_, self.seq_len: l, self.masks: m})

        return pred, embed


    def train(self, state, label):
        state_, l, m = batchToArrays(state)
        _, entrophy = self.session.run([self.optim, self.rho_mean], feed_dict={self.state: state_, self.seq_len: l, self.masks: m, self.label: label})
        return entrophy


    def network(self, state, mask, emb_layer_sizes = [2,64,EMBEDDING_LAYER_SIZE], net_layer_sizes = [EMBEDDING_LAYER_SIZE,64,3]):
    # This could probably be moved into a 'models' file
        d = net_layer_sizes ; d_e = emb_layer_sizes
        num_layers = len(d)-1
        num_layers_e = len(d_e)-1

        # Set up params
        with tf.variable_scope("params") as vs:

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
        #embed_ = tf.nn.max_pool([embeds], ksize=[1, 1, 1000, 1], strides=[1, 1, 1000, 1], padding="SAME")
        #embed = tf.reshape(embed_, [-1, 256])
        embed = tf.reduce_sum(embeds, 1) / tf.reduce_sum(mask, 1)

        # Prediction network
        fc = embed
        for i in range(num_layers-1):
            fc = tf.nn.relu(tf.matmul(fc, w_n[i]) + b_n[i])
        
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

        # Returns the output Q-values
        return predict, tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name), rho, embed


    def rnn(self, state, seq_len, d = [2,32,32,3]):
        num_layers = len(d)-2

        # Build graph
        lstm_cells = []
        for i in range(num_layers): lstm_cells.append(rnn_cell.LSTMCell(d[i+1]))
        multi_cell = rnn_cell.MultiRNNCell(lstm_cells)

        with tf.variable_scope("params") as vs:
          w = tf.Variable(tf.random_normal((d[-2],d[-1]), stddev=0.1), name='w')
          b = tf.Variable(tf.zeros(d[-1]), name='b')
          output, _ = tf.nn.dynamic_rnn(multi_cell, state, sequence_length = seq_len, dtype=tf.float32)

        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, tf.shape(output)[0] - 1)
        prediction = tf.nn.softmax( tf.matmul(last, w) + b )

        return prediction, tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name), last, last


def batchToArrays(input_list):
    max_len = 0
    out = []; seq_len = []; masks = []
    for i in input_list: max_len = max(len(i),max_len)
    for l in input_list:
        out.append(np.pad(np.array(l,dtype=np.float32), ((0,max_len-len(l)),(0,0)), mode='constant'))
        seq_len.append(len(l))
        masks.append(np.pad(np.array(np.ones((len(l),EMBEDDING_LAYER_SIZE)),dtype=np.float32), ((0,max_len-len(l)),(0,0)), mode='constant'))
    return out, seq_len, masks


class shapeGenerator():
    def __init__(self, seed):
        np.random.seed(seed)

    def getPoint(self, shape_type, percent):
      if shape_type == 0: #"circle"
        x = np.cos(percent * 2 * np.pi)
        y = np.sin(percent * 2 * np.pi)
        return x,y

      if shape_type == 1: #"square"
        # Split into 4 lines, renormalise percentage, and return position along line      
        if percent < 0.25:
          per = percent*4
          x = 2*per-1 ; y = -1
        elif percent < 0.5:
          per = (percent-0.25)*4
          x = 1 ; y = 2*per-1
        elif percent < 0.75:
          per = (percent-0.5)*4
          x = 1-2*per ; y = 1
        else:
          per = (percent-0.75)*4
          x = -1 ; y = 1-2*per
        return x,y

      if shape_type == 2: #"triangle"
        # Split into 3 lines, renormalise percentage, and return position along line, then scale to a radius of 1
        if percent < 0.3333:
          per = percent*3
          x = per-0.5 ; y = -0.2887
        elif percent < 0.6666:
          per = (percent-0.3333)*3
          x = 0.5-0.5*per ; y = -0.2887+0.866*per
        else:
          per = (percent-0.6666)*3
          x = -0.5*per ; y = 0.5774-0.866*per
        #x *= 1.73 ; y *= 1.73
        x *= 2 ; y *= 2
        return x,y


    def getShape(self):
        shape = []
        N = 10 + np.random.randint(10) #Number of points
        NOISE_SCALE = 0.01

        shape_type = np.random.randint(3) # type of shape
        scale = 0.4 + np.random.random()*0.6 # scale factor
        rot = np.random.random() # rotation factor
        cos_rot = np.cos(rot * 2 * np.pi)
        sin_rot = np.sin(rot * 2 * np.pi)

        noise_x = 2*np.random.rand(N)-1 # x noise for each point
        noise_y = 2*np.random.rand(N)-1 # y noise for each point
        pos = np.random.rand(N) # position of each point along the outline of the shape

        x_shift = np.random.random()*0.3
        y_shift = np.random.random()*0.3

        for i in range(N):
          x, y = self.getPoint(shape_type, pos[i])
          x += noise_x[i]*NOISE_SCALE ; y += noise_y[i]*NOISE_SCALE 
          x *= scale ; y *= scale

          x_ = x*cos_rot - y*sin_rot
          y_ = x*sin_rot + y*cos_rot
          x_ += x_shift ; y_ += y_shift 
          shape.append([x_,y_])


        label = np.zeros(3) ; label[shape_type] = 1
        data = np.array([scale, x_shift, y_shift])
        return shape, label, data
            
    def getPerfectShape(self):
        shape = []
        N = 10 + np.random.randint(10) #Number of points

        shape_type = np.random.randint(3) # type of shape
        scale = 0.4 + np.random.random()*0.6 # scale factor
        rot = np.random.random() # rotation factor
        cos_rot = np.cos(rot * 2 * np.pi)
        sin_rot = np.sin(rot * 2 * np.pi)

        for i in range(N):
          x, y = self.getPoint(shape_type, i / N)
          x *= scale ; y *= scale
          x_ = x*cos_rot - y*sin_rot
          y_ = x*sin_rot + y*cos_rot
          shape.append([x_,y_])

        label = np.zeros(3) ; label[shape_type] = 1
        data = np.array([1, 0, 0])
        return shape, label, data


    def getBatch(self, batch_size):
        shapes = []
        labels = []
        metadata = []
	for i in range(batch_size):
            shape, label, data = self.getShape()
            shapes.append(shape) ; labels.append(label) ; metadata.append(data)
        return shapes, labels, metadata
        


def main(_):
  # Launch the graph
  with tf.Session() as sess:

    training_iters = args.training_iters
    display_step = args.display_step
    save_step = display_step*5
    batch_size = args.batch_size

    args.input_size = 2
    args.num_actions = 3

    # Set up agent and graph
    agent = Agent(sess, args)

    # Load saver after agent tf variables initialised
    saver = tf.train.Saver()

    # Load or initialise variables
    if args.load_from is not None:
      # Load from file
      ckpt = tf.train.get_checkpoint_state(args.load_from)
      print("Loading model from {}".format(ckpt.model_checkpoint_path))
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      # Initialize the variables
      sess.run(tf.initialize_all_variables())

    # Seed Environment
    env = shapeGenerator(123)

    # Inititalise statistics
    l = 0

    # Keep training until reach max iterations
    for step in tqdm(range(training_iters), ncols=70):

      # Train 
      state, label, metadata = env.getBatch(64)
      loss = agent.train(state, label)

      # Update Statistics
      l += loss
 
      # Display Statistics
      if (step) % display_step == 0:
           # Calculate validation
           state, label, _ = env.getBatch(1000)
           pred, emb = agent.predict(state)
           pred_hot = np.zeros_like(pred) ; pred_hot[pred == pred.max(axis=1)[:,None]] = 1
           right = np.sum(pred_hot * label)

           e_mean = np.mean(emb)*100

           # Visualise
           if step >= 5000:
             np.set_printoptions(threshold='nan', precision=3, suppress=True)
             shape, label, meta = env.getShape()

             pred, e = agent.predict([shape])
             shape_ = np.transpose(shape)
             plt.scatter(shape_[0], shape_[1], 40)
             plt.scatter(np.arange(EMBEDDING_LAYER_SIZE)*(2/EMBEDDING_LAYER_SIZE) - 0.5, np.full(EMBEDDING_LAYER_SIZE,-1), 20, e[0], 's')
             plt.text(-0.9, -6, e)
             plt.text(-0.9, -2.3, pred[0])
             plt.text(-0.9, -2.5, label)
             plt.scatter([-1,2,-1,2], [-0.7,-0.7,1.7,1.7], 0)
             plt.show()

             shape, label, meta = env.getPerfectShape()
             pred, e = agent.predict([shape])
             shape_ = np.transpose(shape)
             plt.scatter(shape_[0], shape_[1], 40)
             plt.text(-0.9, -0.1, pred[0])
             plt.text(-0.9, -0.3, label)
             plt.scatter([-1,2,-1,2], [-0.7,-0.7,1.7,1.7], 0)
             plt.show()

           tqdm.write("{}, {:>7}/{}it | l: {:4.2f}, l_m: {:4.2f}, acc: {:4.2f}".format(time.strftime("%H:%M:%S"), step, training_iters, l, e_mean, right))

           state_, l, m = batchToArrays(state)
           num = agent.session.run(agent.KL_p, feed_dict={agent.state: state_, agent.seq_len: l, agent.masks: m})
           #print num

           l = 0

      # Save model
      if ((step+1) % save_step == 0) & (args.save_dir is not None):
          d = os.path.dirname(args.save_dir)
          if not os.path.exists(d):
              os.makedirs(d)
          checkpoint_path = os.path.join(args.save_dir, 'checkpoint.ckpt')
          tqdm.write("Saving model to {}".format(checkpoint_path))
          saver.save(sess, checkpoint_path)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_iters', type=int, default=50000,
                       help='Number of training iterations to run for')
    parser.add_argument('--display_step', type=int, default=100,
                       help='Number of iterations between parameter prints')

    parser.add_argument('--batch_size', type=int, default=32,
                       help='Size of batch for Q-value updates')

    parser.add_argument('--learning_rate', type=float, default=0.0025,
                       help='Learning rate for TD updates')

    parser.add_argument('--save_dir', type=str, default=None,
                       help='data directory to save checkpoints')
    parser.add_argument('--load_from', type=str, default=None,
                       help='Location of checkpoint to resume from')

    args = parser.parse_args()

    #args.layer_sizes = [int(i) for i in (args.layer_sizes.split(',') if args.layer_sizes else [])]

    print args

    tf.app.run()

