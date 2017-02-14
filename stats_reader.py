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
    for summary in summaries:
       offset = 0
       agent_type = summary['agent_params']['agent_type']
       emb_size = summary['agent_params']['embedding_size']
       sparsity_reg = summary['agent_params']['sparsity_reg']

       accuracy = summary['accuracy']

       rhos = summary['rho_mean']
       _, rhos = smooth_set(rhos)
       rho = rhos[-1] ; k = rho*emb_size

       #x_s.append(emb_size) ; y_s.append(k)
       x_s.append(sparsity_reg) ; y_s.append(rho)
       avr_x, avr_y = smooth_set(accuracy)
       if agent_type == 'embed': offset += 10000
       if emb_size == 256: offset += 5000
       plt.plot(avr_x + offset, avr_y)

    #plt.scatter(x_s, y_s, 20)
    plt.show()

def smooth_set(y_val):
    step_size = 100
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


