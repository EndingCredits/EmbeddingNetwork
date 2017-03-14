from __future__ import division

import argparse
import os
import time
from tqdm import tqdm
import pprint

import numpy as np
import tensorflow as tf

from agent import Agent
from ShapesDataset import shapeGenerator

import matplotlib.pyplot as plt

def main(_):
  
  # Launch the graph
  with tf.Session() as sess:

    summaries = []
    filename = args.file

    # Set up agents
    runs = []
    for sd in [ 123, 1234, 12345 ]:#, 123456, 1234567 ]:
     for e in [ 256 ]:
      for n in [ True, False ]:
       for ty in [ 'reembedding' ]:
        agent_params = {
            'agent_type': ty,
            'input_size': 2,
            'num_classes': 3,
            'embedding_size': e,
            'learning_rate': 0.0025,
            'rho_target': 0.05,
            'sparsity_reg': 0.0,
            'seed': sd
        }

        Agent(sess, agent_params)

        env_params = { 'num_points': 15,
                   'point_dist': n,
                   'num_extra_points': 10,
                   'point_noise_scale': 0.1,
                   'shape_noise_scale': 0.5,
                   'scale_min': 0.1,
                   'initial_seed': 1234,
                   'dataset_size': 100000 }

        run = { 'Agent': Agent(sess, agent_params),
                'Env': shapeGenerator(env_params) }
        runs.append(run)

    # Initialise variables
    sess.run(tf.global_variables_initializer())

    for run in runs:
      agent = run['Agent'] ; env = run['Env']
      stats = train_agent(agent, env, 5000)
      test_stats = test_agent(agent, env)

      summary = { 'agent_params': agent.hyperparams, 'env_params': env.params, 'step': stats['step'], 'accuracy': stats['accuracy'],
                  'test_accuracy': test_stats['accuracy'] }
      summaries.append(summary)

      print "Saving statistics to " + filename + "..."
      np.save(filename, summaries)
      print
    

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
          #rho.append(s['rho_mean']) ; emb.append(np.mean(s['embedding']))
          #pq.append(np.mean(s['pq_mean_sq']))
 
      # Display Statistics
      if (step) % display_step == 0:
         l = np.mean(loss[last_update:]) ; a = np.mean(acc[last_update:]) * 100# ; r = np.mean(pq[last_update:])
         tqdm.write("{}, {:>7}/{}it | loss: {:4.2f}, acc: {:4.2f}%".format(time.strftime("%H:%M:%S"), step, training_iters, l, a ))
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
      #pq.append(s['pq_mean_sq'])

    loss_ = np.mean(loss) ; acc_ = np.mean(acc)
    stats = { 'accuracy': acc_, 'loss': loss_ }

    return stats


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='stats.npy',
                       help='Filename to save statistice to.')

    args = parser.parse_args()

    tf.app.run()

