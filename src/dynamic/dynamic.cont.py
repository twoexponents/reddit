import tensorflow as tf
import sys
from mydynamiclib import runRNNModel

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

hidden_size = 16#32#64#32
learning_rate = 0.001
batch_size = 128#128#64 #32
epochs = 100
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
    for seq_length in range(input_length, input_length+1):
        runRNNModel(hidden_size, learning_rate, batch_size, epochs, keep_rate, seq_length=seq_length, exclude_newbie=exclude_newbie, cont=1)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv])

