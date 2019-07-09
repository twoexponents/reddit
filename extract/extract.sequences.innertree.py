
import MySQLdb


import cPickle as pickle

import random


def sql_connect():
  conn = MySQLdb.connect(host="localhost", user="root", passwd="mmlab", db="reddit_news")
  cursor = conn.cursor()
  return conn, cursor


def sql_close(cursor, conn):
  cursor.close()
  conn.close()


if __name__ == '__main__':
  import sys, os

  s_leaf = pickle.load(open('../data/leaf.p', 'r'))

  d_files = {}

  for i in xrange(1, 21):
    key = 'learn.%d'%(i)
    d_files[key] = open('/home/jhlim/data/seq.%s.csv'%(key), 'w')

    key = 'test.%d'%(i)
    d_files[key] = open('/home/jhlim/data/seq.%s.csv'%(key), 'w')

  f = open('../data/sequences.csv', 'r')
  lines = f.readlines()
  f.close()

  d_trees = {}
  for line in lines:
      line = line.replace('\n', '')
      post_key = line.split(',')[0]
      if post_key not in d_trees:
          d_trees[post_key] = []
      d_trees[post_key].append(line)

  for target_seq in xrange(1, 21):
    s = set()
    for tree in d_trees:
      valid_seq = []
      for seq in d_trees[tree]:
        elements = seq.split(',')
        
        if len(elements) < target_seq:
          continue
        
        seqs = elements[:target_seq]
        if seqs[-1] in s:
          continue
        valid_seq.append(seqs) #seqs = ['t3_a', 't1_b', 't1_c']

      learn_len = (int) (len(valid_seq) * 4 / 5)
      
      if len(valid_seq) == 1:
        if random.random() < 0.8:
          learn_len = 1

      for seqs in valid_seq[:learn_len]:
        s.add(seqs[-1])
        b_leaf = seqs[-1] in s_leaf

        key = 'learn.%d'%(target_seq)
        d_files[key].write('%s,%d\n'%(','.join(seqs), b_leaf))

      for seqs in valid_seq[learn_len:]:
        s.add(seqs[-1])
        b_leaf = seqs[-1] in s_leaf

        key = 'test.%d'%(target_seq)
        d_files[key].write('%s,%d\n'%(','.join(seqs), b_leaf))
        

  for f in d_files.values():
    f.close()
