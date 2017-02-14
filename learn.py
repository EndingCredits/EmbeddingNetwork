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
    agents = []
    for ty in [ 'reembedding' ]: 
      for es in [ 256 ]: #[ 1024, 768, 512, 384, 256, 192, 128, 96, 64 ]:
        for lr in [ 0.0025 ]:
          for sr in [ 0.0 ]:#, 0.001, 0.0025, 0.005, 0.01 ]:
              agent_params = {
                  'agent_type': ty,
                  'input_size': 2,
                  'num_classes': 3,
                  'embedding_size': es,
                  'learning_rate': lr,
                  'rho_target': 0.05,
                  'sparsity_reg': sr
              }
              agents.append(Agent(sess, agent_params))

    # Initialise variables
    sess.run(tf.global_variables_initializer())

    # Seed Environment
    env_params = { 'num_points': 10,
                   'num_extra_points': 10,
                   'point_noise_scale': 0.2,
                   'shape_noise_scale': 2.0,
                   'scale_min': 0.02 }

    for agent in agents:
      env = shapeGenerator(123, env_params)
      stats = train_agent(agent, env, 10000)
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
    loss = [] ; acc = [] ; rho = [] ; emb = []

    # Keep training until reach max iterations
    for step in tqdm(range(training_iters), ncols=70):

      # Train 
      state, label, metadata = env.getBatch(64)
      l, a, s = agent.train(state, label)

      # Update Statistics
      loss.append(l) ; acc.append(a) ; rho.append(s['rho_mean']) ; emb.append(np.mean(s['embedding']))
 
      # Display Statistics
      if (step) % display_step == 0:
         l = np.mean(loss[-display_step:]) ; a = np.mean(acc[-display_step:]) * 100
         tqdm.write("{}, {:>7}/{}it | loss: {:4.2f}, acc: {:4.2f}%".format(time.strftime("%H:%M:%S"), step, training_iters, l, a))       

    stats = { 'accuracy': acc, 'loss': loss, 'rho_mean': rho, 'embedding_mean': emb }

    return stats


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='stats.npy',
                       help='Filename to save statistice to.')

    args = parser.parse_args()

    tf.app.run()

