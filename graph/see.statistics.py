import pymysql
import pickle
from statistics import mean
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from operator import itemgetter

import re

def sql_connect():
  conn = pymysql.connect(host="localhost", user="root", passwd="mmlab2019", db="reddit_news")
  cursor = conn.cursor()
  return conn, cursor

def sql_close(cursor, conn):
  cursor.close()
  conn.close()


if __name__ == '__main__':
  sentences = pickle.load(open('/home/jhlim/data/lastcommentbodyfeatures.p', 'rb'))
  #sentences = pickle.load(open('/home/jhlim/data/commentbodyfeatures.p', 'rb'))
  conn, cursor, = sql_connect()

  d = {}

  sql = """
    SELECT comment_key, body, is_valid, author, UNIX_TIMESTAMP(created_utc)
    FROM comments
    """
  cursor.execute(sql)
  rs = cursor.fetchall()

  for item in rs:
    comment_key, text, is_valid, author, ts,  = item

    if comment_key not in d:
        d[comment_key] = {}
        d[comment_key]['valid'] = is_valid
        d[comment_key]['author'] = author
        d[comment_key]['timestamp'] = ts

  sql = """
    SELECT post_key, title, author, UNIX_TIMESTAMP(created_utc)
    FROM posts
    """
  cursor.execute(sql)
  rs = cursor.fetchall()

  for item in rs:
    post_key, text, author, ts, = item

    if post_key not in d:
        d[post_key] = {}
        d[post_key]['valid'] = 1
        d[post_key]['author'] = author
        d[post_key]['timestamp'] = ts


  no_valid = 0; no_author = 0;
  f = open('/home/jhlim/SequencePrediction/data/sequences.csv', 'r')
  lines = f.readlines()
  
  max_ts = 0; max_post_ts = 0
  posts = {}; comments = {}; users = {}; comment_lens = {}; comment_words = {};
  for line in lines:
      items = line.replace('\n', '').split(',')
      for i, item in enumerate(items):
          if i == 0 and item not in posts:
              posts[item] = 1
          elif i != 0 and item not in comments:
              comments[item] = 1

          if item in d:
              user = d[item]['author']
              if user not in users:
                  users[user] = 1

          if i != 0 and item in sentences:
              if item not in comment_lens:
                  comment_lens[item] = len(sentences[item])
                  comment_words[item] = len(sentences[item].split(' '))


  print ('seqs: %d, no_valid: %d'%(len(lines), no_valid))
  print ('# posts: %d, # comments: %d, # users: %d'%(len(posts), len(comments), len(users)))
  print ('avg comment length: %f, min: %f, max: %f'%(mean(comment_lens.values()), min(comment_lens.values()), max(comment_lens.values())))
  print ('avg comment word length: %f, min: %f, max: %f'%(mean(comment_words.values()), min(comment_words.values()), max(comment_words.values())))


  sql_close(cursor, conn)


