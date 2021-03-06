import tensorflow as tf
import sys
from typefeatureslib import runRNNModel

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

hidden_size = 128 #16
learning_rate = 0.001
batch_size = 128
epochs = 100
keep_rate = 0.5

def main(argv):
    features = []
    input_length = 1
    for i, argv in enumerate(sys.argv[1:]):
        if i == 0:
            input_length = int(argv)
        else:
            features.append(argv)

    print ('sequence len: %d' % (input_length))
    print ('learning_rate: %f, batch_size %d, epochs %d' % (learning_rate, batch_size, epochs))

    # 1.1 load feature dataset
    for seq_length in range(input_length, input_length+1):
        runRNNModel(hidden_size, learning_rate, batch_size, epochs, keep_rate, features, seq_length=seq_length)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv])

