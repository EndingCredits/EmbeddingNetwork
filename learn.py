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


def main(_):
  
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    summaries = []
    filename = args.file

    # Set up agent
    runs = []
    for sd in [ 123, 123456, 1234567 ]:
        agent_params = {
            'agent_type': 'default',
            'input_size': 2,
            'num_classes': 3,
            'embedding_size': 256
        }

        env_params = { }

        run = { 'agent_params': agent_params,
                'env_params': env_params }
        runs.append(run)


    for run in runs:
      # Launch the graph
      with tf.Session(config=config) as sess:
        agent_params = run['agent_params'] ; env_params = run['env_params']
          
        print "Building agent with params: "
        print agent_params
        agent = Agent(sess, agent_params)
        print "Building environment with params: "
        print env_params
        env = shapeGenerator(env_params)
        sess.run(tf.global_variables_initializer())
          
        stats = train_agent(agent, env, 1000)
        test_stats = test_agent(agent, env)

        summary = { 'agent_params': agent.hyperparams, 'env_params': env.params, 'step': stats['step'], 'accuracy': stats['accuracy'],
                      'test_accuracy': test_stats['accuracy'] }
        summaries.append(summary)

        print "Saving statistics to " + filename + "..."
        np.save(filename, summaries)
        print
        #tf.reset_default_graph()
    

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
 
      # Display Statistics
      if (step) % display_step == 0:
         l = np.mean(loss[last_update:]) ; a = np.mean(acc[last_update:]) * 100
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

    loss_ = np.mean(loss) ; acc_ = np.mean(acc)
    stats = { 'accuracy': acc_, 'loss': loss_ }

    return stats


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='results.npy',
                       help='Filename to save statistics to.')

    args = parser.parse_args()

    tf.app.run()

