from __future__ import division

import os
import time
from tqdm import tqdm
import pprint

import numpy as np
import tensorflow as tf

from agent import Agent
from ModelDataset import ModelDataset


def main(_):
  
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True

  # Launch the graph
  with tf.Session(config=config) as sess:

    # Set up agents
    agent_params = {
        'agent_type': 'pointnet',
        'input_size': 3,
        'num_classes': 40,
        'learning_rate': 0.0001,
        'optimizer': 'adamax'
    }

    Agent(sess, agent_params)

    env_params = { 'train_file': 'data/ModelNet40train_100.npy',
                   'test_file': 'data/ModelNet40test_100.npy',
                   'seed': 1234 }

    agent = Agent(sess, agent_params)
    env = ModelDataset(env_params)

    # Initialise variables
    sess.run(tf.global_variables_initializer())

    train_agent(agent, env, 10000)
    test_agent(agent, env)

    

def train_agent(agent, env, training_iters, display_step = 100):

    print "Training agent with params: "
    for key in agent.hyperparams.keys():
        print "    " + key + ": " + str(agent.hyperparams[key])

    # Inititalise statistics
    steps = [0] ; loss = [ 0.0, ] ; acc = [ 0.0, ]
    train_loss = [ 0.0, ] ; train_acc = [ 0.0, ]
    last_update = 0 ; last_update_train = 0

    # Keep training until reach max iterations
    for step in tqdm(range(training_iters), ncols=70):

      # Train
      state, label, metadata = env.getBatch(64)
      l, a, _ = agent.train(state, label)
      train_loss.append(l) ; train_acc.append(a)

      if (step) % 10 == 0:
          state, label, metadata = env.getBatch(64, True)
          l, a, s = agent.test(state, label)

          # Update Statistics
          steps.append(step) ; loss.append(l) ; acc.append(a)
 
      # Display Statistics
      if (step) % display_step == 0:
         l = np.mean(loss[last_update:]) ; a = np.mean(acc[last_update:]) * 100
         l_ = np.mean(train_loss[last_update_train:]) ; a_ = np.mean(train_acc[last_update_train:]) * 100
         tqdm.write("{}, {:>7}/{}it | train_loss: {:4.2f}, train_acc: {:4.2f}%, test_loss: {:4.2f}, test_acc: {:4.2f}%".format(
             time.strftime("%H:%M:%S"), step, training_iters, l_, a_, l, a))
         last_update = np.size(loss) ; last_update_train = np.size(train_loss)

    return 0


def test_agent(agent, env, test_iters=100):

    print "Testing: "

    # Inititalise statistics
    loss = [ ] ; acc = [ ] ; pq = [ ]

    for step in tqdm(range(test_iters), ncols=70):

      # Train
      state, label, metadata = env.getBatch(64, True)
      l, a, s = agent.test(state, label)
      loss.append(l) ; acc.append(a)

    loss_ = np.mean(loss) ; acc_ = np.mean(acc)
    print "Test accuracy: " + str(acc_)

    return 0


if __name__ == '__main__':

    tf.app.run()
