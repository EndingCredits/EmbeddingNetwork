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
    for sd in [ 123, 1337, 13579 ]:
      for ty in [ 'embed', 'reembedding' ]:
        agent_params = {
            'agent_type': ty,
            'input_size': 2,
            'num_classes': 3,
            'embedding_size': 128,
            'learning_rate': 0.0025,
            'rho_target': 0.05,
            'sparsity_reg': 0.0,
            'seed': sd
        }

        Agent(sess, agent_params)

        env_params = { 'num_points': 10,
                   'num_extra_points': 10,
                   'point_noise_scale': 0.2,
                   'shape_noise_scale': 1.0,
                   'scale_min': 0.1,
                   'initial_seed': 1234,
                   'dataset_size': 10000 }

        run = { 'Agent': Agent(sess, agent_params),
                'Env': shapeGenerator(env_params) }
        runs.append(run)

    # Initialise variables
    sess.run(tf.global_variables_initializer())

    for run in runs:
      agent = run['Agent'] ; env = run['Env']
      stats = train_agent(agent, env, 5000)

      summary = { 'agent_params': agent.hyperparams, 'env_params': env.params, 'accuracy': stats['accuracy'], 'rho_mean': stats['rho_mean'], 'embedding_mean': stats['embedding_mean']  }
      summaries.append(summary)

      print "Saving statistics to " + filename + "..."
      np.save(filename, summaries)
      print
    

def train_agent(agent, env, training_iters, display_step = 100):

    print "Training agent with params: "
    for key in agent.hyperparams.keys():
        print "    " + key + ": " + str(agent.hyperparams[key])

    # Inititalise statistics
    loss = [ 0.0, ] ; acc = [ 0.0, ] ; rho = [] ; emb = []
    last_update = 0

    # Keep training until reach max iterations
    for step in tqdm(range(training_iters), ncols=70):

      # Train
      state, label, metadata = env.getBatch(64)
      agent.train(state, label)

      if (step) % 10 == 0:
          state, label, metadata = env.getBatch(100, True)
          l, a, s = agent.test(state, label)

          # Update Statistics
          loss.append(l) ; acc.append(a) ; rho.append(s['rho_mean']) ; emb.append(np.mean(s['embedding']))
 
      # Display Statistics
      if (step) % display_step == 0:
         l = np.mean(loss[last_update:]) ; a = np.mean(acc[last_update:]) * 100
         tqdm.write("{}, {:>7}/{}it | loss: {:4.2f}, acc: {:4.2f}%".format(time.strftime("%H:%M:%S"), step, training_iters, l, a))
         last_update = np.size(loss)

    stats = { 'accuracy': acc, 'loss': loss, 'rho_mean': rho, 'embedding_mean': emb }

    return stats


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='stats.npy',
                       help='Filename to save statistice to.')

    args = parser.parse_args()

    tf.app.run()

