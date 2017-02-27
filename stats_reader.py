from __future__ import division

import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():

    filename = args.file
    print "Loading from " + filename + "..."
    summaries = np.load(filename)

    np.set_printoptions(threshold='nan', precision=3, suppress=True)
    
    x_s = [] ; y_s = []
    c_s = [] #colours
    for summary in summaries:
       offset = 0
       agent_type = summary['agent_params']['agent_type']
       emb_size = summary['agent_params']['embedding_size']
       sparsity_reg = summary['agent_params']['sparsity_reg']

       num_points = summary['env_params']['num_points']

       accuracy = summary['accuracy']
       test_accuracy = summary['test_accuracy']

       #rhos = summary['rho_mean']
       #_, rhos = smooth_set(rhos)
       #rho = rhos[-1] ; k = rho*emb_size

       pq = summary['pq']
       pq_ = (pq*emb_size)
       pq_pred = emb_size / float(num_points)

       #x_s.append(emb_size) ; y_s.append(k)
       #x_s.append(sparsity_reg) ; y_s.append(rho)
       x_s.append(num_points) ; y_s.append(pq)
       #x_s.append(num_points) ; y_s.append(pq_)

       c = 'm'
       if emb_size == 32: c = 'b'
       if emb_size == 64: c = 'r'
       if emb_size == 128: c = 'm'
       if emb_size == 256: c = 'y'

       c_ = 'm'
       if num_points == 5: c_ = 'b'
       if num_points == 10: c_ = 'r'
       if num_points == 20: c_ = 'm'
       if num_points == 40: c_ = 'y'

       c_s.append(c)

       avr_x, avr_y = smooth_set(accuracy)

       colour = 'm'
       if agent_type == 'embed': colour = 'b'
       if agent_type == 'RNN': colour = 'r'
       if agent_type == 'simple': colour = 'g'
       if agent_type == 'reembedding': colour = 'y'

       #plt.plot(avr_x + offset, avr_y, color=colour)
       #if agent_type == 'RNN': plt.scatter(100, test_accuracy, 20)
       #if agent_type == 'embed': plt.scatter(200, test_accuracy, 20)

    plt.scatter(x_s, y_s, 40, color = c_s)
    plt.show()

def smooth_set(y_val):
    step_size = 20
    total = np.size(y_val)
    num = int(total/step_size)
    avr_x = np.zeros(num)
    avr_y = np.zeros(num)
    for i in range(num):
        avr_x[i] = (i+0.5)*step_size
        avr_y[i] = np.mean(y_val[i*step_size:(i+1)*step_size])
    return avr_x, avr_y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='stats.npy',
                       help='Filename to save statistice to.')

    args = parser.parse_args()

    main()


