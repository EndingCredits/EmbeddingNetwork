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
        'agent_type': 'PCL',
        'input_size': 3,
        'num_classes': 40,
        'embedding_size': 256,
        'learning_rate': 0.0005,
        'rho_target': 0.05,
        'sparsity_reg': 0.0,
        'seed': 123
    }

    Agent(sess, agent_params)

    env_params = { 'train_file': 'train.npy',
                   'test_file': 'test.npy',
                   'seed': 1234 }

    agent = Agent(sess, agent_params)
    env = ModelDataset(env_params)

    print env.size

    # Initialise variables
    sess.run(tf.global_variables_initializer())

    stats = train_agent(agent, env, 100000)
    test_stats = test_agent(agent, env)

    

def train_agent(agent, env, training_iters, display_step = 100):

    print "Training agent with params: "
    for key in agent.hyperparams.keys():
        print "    " + key + ": " + str(agent.hyperparams[key])

    # Inititalise statistics
    steps= [0] ; loss = [ 0.0, ] ; acc = [ 0.0, ]
    rho = [] ; emb = [] ; pq = []
    last_update = 0

    # Keep training until reach max iterations
    for step in tqdm(range(training_iters), ncols=70):

      # Train
      state, label, metadata = env.getBatch(64)
      agent.train(state, label)

      if (step) % 10 == 0:
          state, label, metadata = env.getBatch(64, True)
          l, a, s = agent.test(state, label)

          # Update Statistics
          steps.append(step) ; loss.append(l) ; acc.append(a)
          rho.append(s['rho_mean']) ; emb.append(np.mean(s['embedding']))
          pq.append(np.mean(s['pq_mean_sq']))
 
      # Display Statistics
      if (step) % display_step == 0:
         l = np.mean(loss[last_update:]) ; a = np.mean(acc[last_update:]) * 100 ; r = np.mean(pq[last_update:])
         tqdm.write("{}, {:>7}/{}it | loss: {:4.2f}, acc: {:4.2f}%, pq: {:4.2f}".format(time.strftime("%H:%M:%S"), step, training_iters, l, a, r))
         last_update = np.size(loss)

    stats = { 'step': steps, 'accuracy': acc, 'loss': loss, 'pq_mean': pq }

    return stats


def test_agent(agent, env, test_iters=100):

    print "Testing: "

    # Inititalise statistics
    loss = [ ] ; acc = [ ] ; pq = [ ]

    for step in tqdm(range(test_iters), ncols=70):

      # Train
      state, label, metadata = env.getBatch(64, True)
      l, a, s = agent.test(state, label)
      loss.append(l) ; acc.append(a)
      pq.append(s['pq_mean_sq'])

    loss_ = np.mean(loss) ; acc_ = np.mean(acc) ; pq_ = np.mean(pq)
    stats = { 'accuracy': acc_, 'loss': loss_, 'pq_sq': pq_ }

    print "Test accuracy: " + str(acc_)

    return stats


if __name__ == '__main__':

    tf.app.run()
