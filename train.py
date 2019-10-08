from __future__ import division

import argparse
import os
import time
from tqdm import tqdm
import pprint

import numpy as np
import tensorflow as tf

from ShapesDataset import shapeGenerator
from ModelDataset import ModelDataset
from PokerDataset import PokerDataset

import networks
from layers import linear, normalize


def main(_):
    batch_size = 64
    learning_rate = 0.00025
    training_iters = 100000
    
    #np.set_printoptions(threshold='nan', precision=3, suppress=True)
    
    env_params = {
        'num_points': 20,
        'num_extra_points': 0,
        'point_noise_scale': 0.2,
        'shape_noise_scale': 0.0, #0.0-2.0
        'scale_min': 0.5,
        'uniform_point_distribution': False,
        'dataset_size': 10000
    }

    env_params_modelnet = {
        'train_file': './data/ModelNet40train_100.npy',
        'test_file': './data/ModelNet40test_100.npy',
        'seed': 123,
    }

    env_fn = lambda: shapeGenerator(env_params)

    model_fn = lambda x: networks.deep_sets_network(x, use_equivariant=True,
    embedding_layers = [128,128,128], output_layers = [128],
    pool_type='max')

    #x = normalize(x)


    #model_fn = networks.statistic_network
    #model_fn = networks.deep_sets_network
    #model_fn = networks.object_embedding_network
    #model_fn = networks.RNN
    #model_fn = networks.pseudo_relation_network
    #model_fn = networks.relation_network
    #model_fn = networks.pointnet
    #model_fn = networks.kary_network

    run(env_fn, model_fn)

def main(_):
    batch_size = 64
    learning_rate = 0.00025
    training_iters = 250000
    
    #np.set_printoptions(threshold='nan', precision=3, suppress=True)

    for emb_width in [ 32, 64, 128 ]:
      for pool_type in [ 'max', 'mean' ]:
        for use_equivariant in [ True, False ]:
          for scale_min in [ 1.0, 0.5, 0.25, 0.1 ]:
          #for shape_noise_scale in [ 0.0, 1.0, 2.0, 4.0 ]:
      
            train_params = {
                'batch_size': 64,
                'learning_rate': 0.00025,
            }

            env_params = {
                'num_points': 20,
                'num_extra_points': 0,
                'point_noise_scale': 0.2,
                'shape_noise_scale': 1.0,
                'scale_min': scale_min,
                'uniform_point_distribution': False,
                'dataset_size': 10000
            }

            env_params_modelnet = {
                'train_file': './data/ModelNet40train_100.npy',
                'test_file': './data/ModelNet40test_100.npy',
                'seed': 123,
            }

            env_fn = lambda: shapeGenerator(env_params)

            model_params = {
                'use_equivariant': use_equivariant,
                'embedding_layers': [emb_width]*3,
                'output_layers': [emb_width],
                'pool_type': pool_type
            }
            model_fn = lambda x: networks.deep_sets_network(x, **model_params)

            #model_params = {
            #    'key_size': emb_width // 2,
            #    'value_size': emb_width,
            #}
            #model_fn = lambda x: networks.relation_network(x, **model_params)

            #x = normalize(x)
            
            with open("results.txt", 'a') as f:
                f.write("{}\n{}\n{}\n".format(env_params, model_params, train_params))
                run(env_fn, model_fn, **train_params, 
                    training_iters=training_iters, out_file=f)

                f.write("\n\n")
            tf.reset_default_graph()
   


def run(env_fn, model_fn,
        batch_size = 64,
        learning_rate = 0.00025,
        training_iters = 250000,
        out_file=None):

    print("Building environment with params: ")
    env = env_fn()
    print(env.params)

    num_input = env.n_inputs
    num_classes = env.n_outputs

    # Launch the graph
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:

        print("Building graph...")

        x = tf.placeholder("float", [None, None, num_input], name="data")
        y = tf.placeholder("float", [None, num_classes], name="label")

        model_out = model_fn(x)
        y_pred = linear(model_out, num_classes, name="class_outputs")

        num_vars = 0
        for v in tf.global_variables():
            shape = "x".join(["{:>3}".format(str(x)) for x in v.get_shape()])
            print("{:<45} {}".format(v.name+',', shape))
            num_vars += np.prod(v.get_shape().as_list())

        if num_vars > 10**6:
            var_str = "{:.3f}M".format(num_vars/10**6)
        else:
            var_str = "{:.3f}K".format(num_vars/10**3)
        print("Total number of variables: " + var_str)

        correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

        
        #loss = tf.nn.softmax_cross_entropy_with_logits( labels = y,
        #                                                logits = y_pred )
        loss = tf.reduce_mean(-tf.reduce_sum(
                y * tf.log(tf.clip_by_value(tf.nn.softmax(y_pred),1e-10,1.0)),
                reduction_indices=[1]
            ))

        optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        sess.run(tf.global_variables_initializer())

        
        display_step = 2500

        train_losses = [] ; train_acc = [] ; last_update_train = 0
        test_losses = [] ; test_acc = [] ; last_update = 0
        # Keep training until reach max iterations
        for step in tqdm(range(training_iters+1), ncols=70):

            # Train
            data, label, metadata = env.getBatch(batch_size)

            _, l, a = sess.run([optim, loss, accuracy],
                feed_dict = { x: batchToArrays(data), y: label })
            train_losses.append(np.mean(l)) ; train_acc.append(np.mean(a))

            if (step) % 10 == 0:
              data, label, metadata = env.getBatch(100, True)
              l, a = sess.run([loss, accuracy],
                feed_dict = { x: batchToArrays(data), y: label })

              # Update Statistics
              test_losses.append(np.mean(l)) ; test_acc.append(np.mean(a))
     
            # Display Statistics
            if (step) % display_step == 0:
                l = np.mean(train_losses[last_update_train:])
                a = np.mean(train_acc[last_update_train:]) * 100
                l_ = np.mean(test_losses[last_update:])
                a_ = np.mean(test_acc[last_update:]) * 100
                tqdm.write("{}, {:>7}/{}it | loss: {:4.2f}, acc: {:4.2f}% |".format(
                    time.strftime("%H:%M:%S"), step, training_iters, l, a ) +\
                    " test_loss: {:4.2f}, test_acc: {:4.2f}%".format(l_, a_))
                last_update_train = np.size(train_losses)
                last_update = np.size(test_losses)
                
                if out_file is not None:
                    out_file.write("{}, {:>7}/{}it | loss: {:4.2f}, acc: {:4.2f}% |".format(
                    time.strftime("%H:%M:%S"), step, training_iters, l, a ) +\
                    " test_loss: {:4.2f}, test_acc: {:4.2f}%\n".format(l_, a_))


def batchToArrays(input_list):
    # Takes an input list of lists (of vectors), pads each list the length of
    #   the longest list, compiles the list into a single n x m x d array, and
    #   returns a corresponding n x m x 1 mask.
    max_len = 0
    out = []; masks = []
    for i in input_list:
        max_len = max(len(i),max_len)
    for l in input_list:
        # Zero pad output
        out.append(np.pad(np.array(l, dtype=np.float32),
            ((0,max_len-len(l)),(0,0)), mode='constant'))

    out = np.array(out, dtype=np.float32)
    #out = (out - np.mean(out, axis=-2, keepdims=True)) / np.std(out, axis=-2, keepdims=True)
    return out


if __name__ == '__main__':
    tf.app.run()

