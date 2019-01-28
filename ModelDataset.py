from __future__ import division

import numpy as np

class ModelDataset():
    params = { }

    def __init__(self, params):

        self.params.update(params)
        self.train_file = self.params['train_file']
        self.test_file = self.params['test_file']

        print(self.train_file)
        self.train_data = np.load(self.train_file, encoding="bytes")[0]
        self.test_data = np.load(self.test_file, encoding="bytes")[0]
        print(self.train_data.keys())
        self.size = len(self.train_data[b'data'])
        self.size_cv = len(self.test_data[b'data'])

        self.initial_seed = self.params['seed']
        self.initial_seed_cv = self.initial_seed
        self.order = list(range(self.size))
        np.random.shuffle(self.order)
        self.order_cv = list(range(self.size_cv))
        np.random.shuffle(self.order_cv)

        self.num_samples = 0         #Number of samples taken.
        self.num_samples_cv = 0      #Number of samples taken for validation.
        
        self.n_inputs = len(self.test_data[b'data'][0][0])
        self.n_outputs = len(self.test_data[b'labels'][0])


    def elementSize(self):
        return self.n_inputs
    
    def numClasses(self):
        return self.n_outputs

    def getBatch(self, batch_size, validation=False):
        shapes = []
        labels = []
        metadata = []
        for i in range(batch_size):
            shape, label, data = self._get_single(self._get_num(validation),validation)
            shapes.append(shape) ; labels.append(label)
        return shapes, labels, metadata


    def _get_single(self, num, validation=False, truncate=False):
        if validation==False:
            pos = self.order[num]
            initial_shape = self.train_data[b'data'][pos]
            label = self.train_data[b'labels'][pos]

            if truncate: 
                mask = np.random.choice([False, True], len(initial_shape), p=[0.5, 0.5])
                truncated_shape = initial_shape[mask]
            else:
                truncated_shape = initial_shape

            scale = 0.8 + 0.45*np.random.random()
            rot = np.random.random() * 2 * np.pi
            cos_rot = np.cos(rot) ; sin_rot = np.sin(rot)

            rotation_mat = np.matrix([[cos_rot, -sin_rot, 0],
                                     [sin_rot,  cos_rot,  0],
                                     [0,        0,        1]])
            transform_mat = scale*rotation_mat

            shape = np.dot(truncated_shape, transform_mat)

        else:
            pos = self.order_cv[num]
            shape = self.test_data[b'data'][pos]
            label = self.test_data[b'labels'][pos]

        metadata = []
        return shape, label, metadata


    def _get_num(self, validation=False):
        if validation is False:
            num = self.num_samples
            self.num_samples = (self.num_samples + 1) % self.size
            if self.num_samples == 0: np.random.shuffle(self.order)
            return num
        else:
            num = self.num_samples_cv
            self.num_samples_cv = (self.num_samples_cv + 1) % self.size_cv
            return num
