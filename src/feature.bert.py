import tensorflow as tf
import numpy as np
import sys

from mytensorlib import runRNNModel
from myloaddatalib import makeLearnTestSet

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

hidden_size = 16
learning_rate = 0.001
batch_size = 32
epochs = 20
keep_rate = 0.5

def main(argv):

    exclude_newbie = 0; input_length = 1
    if len(sys.argv) >= 3:
        exclude_newbie = int(sys.argv[2])
    if len(sys.argv) >= 2:
        input_length = int(sys.argv[1])
    print ('sequence len: %d' % (input_length))
    print ('learning_rate: %f, batch_size %d, epochs %d' % (learning_rate, batch_size, epochs))
    print ('exclude_newbie: %d'%(exclude_newbie))

    # 1.1 load feature dataset
    print ('features are loaded')

    for seq_length in range(input_length, input_length+1):
        learn_X, learn_Y, test_X, test_Y = makeLearnTestSet(seq_length, bert=1, exclude_newbie=exclude_newbie)

        runRNNModel(seq_length, learn_X, learn_Y, test_X, test_Y, hidden_size, learning_rate, batch_size, epochs, keep_rate)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv])

