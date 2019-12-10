import tensorflow as tf
import sys
from mydynamiclib import runRNNModel

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main(argv):
    exclude_newbie = 0; input_length = 1
    hidden_size = 128#64#32
    learning_rate = 0.005
    batch_size = 128#256#64 #32
    epochs = 30
    keep_rate = 0.5
    seed1 = 10
    seed2 = 30

    if len(sys.argv) > 6:
        input_length, hidden_size, learning_rate, batch_size, keep_rate, seed1, seed2 = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    else:
        input_length = int(sys.argv[1])

    print ('input_length', 'hidden_size', 'learning_rate', 'batch_size', 'keep_rate', 'seed1', 'seed2')
    print (input_length, hidden_size, learning_rate, batch_size, keep_rate, seed1, seed2)
    #print ('sequence len: %d' % (input_length))
    #print ('learning_rate: %f, batch_size %d, epochs %d' % (learning_rate, batch_size, epochs))
    #print ('exclude_newbie: %d'%(exclude_newbie))

    # 1.1 load feature dataset
    for seq_length in range(input_length, input_length+1):
        runRNNModel(hidden_size, learning_rate, batch_size, epochs, keep_rate, seq_length=seq_length, exclude_newbie=exclude_newbie, bert=1, liwc=1, cont=1, time=1, seed1=seed1, seed2=seed2)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv])

