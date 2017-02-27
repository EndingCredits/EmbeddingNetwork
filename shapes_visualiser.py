from __future__ import division

import argparse
import numpy as np
import matplotlib.pyplot as plt

from ShapesDataset import shapeGenerator

def main():

    env_params = { 'num_points': 15,
                   'point_dist': True,
                   'num_extra_points': 10,
                   'point_noise_scale': 0.1,
                   'shape_noise_scale': 0.5,
                   'scale_min': 0.1,
                   'initial_seed': 1234,
                   'dataset_size': 100000
                 }

    env = shapeGenerator(env_params)

    seed = 1234

    for i in xrange(100):
        shape, label, meta = env.getShape(123+i)

        shape_ = np.transpose(shape)
        plt.scatter(shape_[0], shape_[1], 40)
        plt.text(-0.9, -0.3, label)
        plt.scatter([-1,2,-1,2], [-0.7,-0.7,1.7,1.7], 0)
        plt.show()

if __name__ == "__main__":

    main()


