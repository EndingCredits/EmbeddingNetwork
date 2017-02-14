from __future__ import division

import numpy as np

class shapeGenerator():
    def __init__(self, seed, params):
        np.random.seed(seed)
        self.params = params
        self.num_points = params['num_points']
        self.num_extra_points = params['num_extra_points']
        self.point_noise_scale = params['point_noise_scale']
        self.shape_noise_scale = params['shape_noise_scale']
        self.scale_min = params['scale_min']

    def getPoint(self, shape_type, percent):
      if shape_type == 0: #"circle"
        x = np.cos(percent * 2 * np.pi)
        y = np.sin(percent * 2 * np.pi)
        return x,y

      if shape_type == 1: #"square"
        # Split into 4 lines, renormalise percentage, and return position along line      
        if percent < 0.25:
          per = percent*4
          # Bottom left to bottom right
          x = 2*per-1 ; y = -1
        elif percent < 0.5:
          per = (percent-0.25)*4
          # Bottom right to top right
          x = 1 ; y = 2*per-1
        elif percent < 0.75:
          per = (percent-0.5)*4
          # Top right to top left
          x = 1-2*per ; y = 1
        else:
          per = (percent-0.75)*4
          # Top left to bottom left
          x = -1 ; y = 1-2*per
        return x,y

      if shape_type == 2: #"triangle"
        # Split into 3 lines, renormalise percentage, and return position along line, then scale to a radius of 1
        if percent < 0.3333:
          per = percent*3
          # Bottom left to bottom right
          x = per-0.5 ; y = -0.2887
        elif percent < 0.6666:
          per = (percent-0.3333)*3
          # Bottom right to top
          x = 0.5-0.5*per ; y = -0.2887+0.866*per
        else:
          per = (percent-0.6666)*3
          # Top to bottom left
          x = -0.5*per ; y = 0.5774-0.866*per
        #x *= 1.73 ; y *= 1.73
        x *= 2 ; y *= 2
        return x,y


    def getShape(self):
        shape = []

        # Generate shape variables
        shape_type = np.random.randint(3)                              # type of shape
        N = self.num_points + np.random.randint(self.num_extra_points) # number of points
        scale = self.scale_min + np.random.random()*(1-self.scale_min) # scale factor
        rot = np.random.random()                                       # rotation factor
        x_shift = (2*np.random.random()-1)*self.shape_noise_scale      # x noise for shape
        y_shift = (2*np.random.random()-1)*self.shape_noise_scale      # y noise for shape

        # Generate variables for each point
        pos = np.random.rand(N)                                        # position of each point along the outline of the shape
        noise_x = (2*np.random.rand(N)-1)*self.point_noise_scale       # x noise for each point
        noise_y = (2*np.random.rand(N)-1)*self.point_noise_scale       # y noise for each point
         
        # Precalc useful values
        cos_rot = np.cos(rot * 2 * np.pi)
        sin_rot = np.sin(rot * 2 * np.pi)

        for i in range(N):
          # Get point from position and shape type
          x, y = self.getPoint(shape_type, pos[i])

          # Add noise
          x += noise_x[i] ; y += noise_y[i]

          # Scale
          x *= scale ; y *= scale

          # Rotate
          x_ = x*cos_rot - y*sin_rot
          y_ = x*sin_rot + y*cos_rot

          # Translate
          x_ += x_shift ; y_ += y_shift 

          # Add point to shape
          shape.append([x_,y_])

        label = np.zeros(3) ; label[shape_type] = 1
        data = np.array([scale, x_shift, y_shift])
        return shape, label, data
            
    def getPerfectShape(self):
        shape = []

        # Generate shape variables
        shape_type = np.random.randint(3)                              # type of shape
        N = self.num_points + np.random.randint(self.num_extra_points) # number of points
        scale = self.scale_min + np.random.random()*(1-self.scale_min) # scale factor
        rot = np.random.random()                                       # rotation factor
        x_shift = (2*np.random.random()-1)*self.shape_noise_scale      # x noise for shape
        y_shift = (2*np.random.random()-1)*self.shape_noise_scale      # y noise for shape

        # Generate variables for each point
        pos = np.random.rand(N)                                        # position of each point along the outline of the shape
        noise_x = (2*np.random.rand(N)-1)*self.point_noise_scale       # x noise for each point
        noise_y = (2*np.random.rand(N)-1)*self.point_noise_scale       # y noise for each point
         
        # Precalc useful values
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
