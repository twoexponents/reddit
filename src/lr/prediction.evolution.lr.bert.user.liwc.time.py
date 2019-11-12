import numpy as np
import sys
from mylrlib import runLRModel
from myloaddatalib import makeLearnTestSet

def main():
  exclude_newbie = 0; input_length = 1
  if len(sys.argv) >= 3:
    exclude_newbie = int(sys.argv[2])
  if len(sys.argv) >= 2:
    input_length = int(sys.argv[1])
  print ('exclude_newbie: %d'%(exclude_newbie))

  for seq_length in range(1, 3):
    f = open('/home/jhlim/data/seq.learn.%d.csv'%(seq_length), 'r')
    learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
    f.close()

    learn_X, learn_Y, test_X, test_Y = makeLearnTestSet(seq_length, bert=1, user=1, liwc=1, time=1, exclude_newbie=exclude_newbie, lr=1)
    runLRModel(seq_length, learn_X, learn_Y, test_X, test_Y)

if __name__ == '__main__':
  main()

