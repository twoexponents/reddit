
import MySQLdb


import cPickle as pickle

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

import string

from collections import defaultdict

import numpy as np

import time
import heapq
import operator


def sql_connect():
    conn = MySQLdb.connect(host="localhost", user="root", passwd="mmlab", db="reddit_news")
    cursor = conn.cursor()
    return conn, cursor


def sql_close(cursor, conn):
    cursor.close()
    conn.close()

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
N = 5 # Get top 5 words

def clean(doc, wv):
    #punc_free = ''.join(ch for ch in list(doc.lower()) if ch not in exclude)

    #stop_free = [i for i in doc.lower().split() if i not in stop]
    #sentence = ' '.join(stop_free)
    sentence = doc.lower()
    punc_free = ''.join(ch for ch in list(sentence) if ch not in exclude)
    notrained_free = ' '.join(word for word in punc_free .split() if word in wv)

    return notrained_free

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in X if w in self.word2vec]
                or [np.zeros(self.dim)], axis=0)
        ])

class TfidfEmbeddingVectorizer(object):
  def __init__(self, word2vec):
    self.word2vec = word2vec
    self.word2weight = None
    self.dim = len(word2vec.itervalues().next())

  def fit(self, X, y):
    tfidf = TfidfVectorizer(analyzer=lambda x: x)
    tfidf.fit(X)
    # if a word was never seen - it must be at least as infrequent
    # as any of the known words - so the default idf is the max of 
    # known idf's
    max_idf = max(tfidf.idf_)
    self.word2weight = defaultdict(
      lambda: max_idf,
      [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

    return self

  def transform(self, X):
    return np.array([
      np.mean([self.word2vec[w] * self.word2weight[w]
                for w in X if w in self.word2vec] or
                [np.zeros(self.dim)], axis=0)
    ])


if __name__ == '__main__':
    import sys, os

    start_time = time.time()

    f = open('../data/sequences.csv', 'r')
    lines = f.readlines()
    f.close()
    
    wv = KeyedVectors.load_word2vec_format('../data/googlenews/GoogleNews-vectors-negative300.bin',
                                            binary=True)
    w2v = dict(zip(wv.index2word, wv.syn0))
    mean_vectorizer = MeanEmbeddingVectorizer(w2v)

    d_trees = {}
    for line in lines:
        line = line.replace('\n', '')
        post_key = line.split(',')[0]
        if post_key not in d_trees:
            d_trees[post_key] = []
        d_trees[post_key].append(line)

    conn, cursor, = sql_connect()

    d_body = {}

    sql = """
        SELECT post_key, title
        FROM posts
        """
    cursor.execute(sql)
    rs = cursor.fetchall()

    for item in rs:
        post_key, text, = item
        d_body[post_key] = clean(text, wv)

    sql = """
        SELECT comment_key, body
        FROM comments
        WHERE is_valid = 1
        """
    cursor.execute(sql)
    rs = cursor.fetchall()

    for item in rs:
        comment_key, text, = item
        d_body[comment_key] = clean(text, wv)

    print 'Load Successfully'   

    d = {} # w2v of top5 words of the element
    d_topwords = {} # top words list of the element

    print '# trees: %d'%(len(d_trees))
    try:
        for i, tree in enumerate(d_trees):
            if i % 100 == 0:
                print '%dth tree'%(i)
            corpus = []

            # Get all corpus of the conv tree
            for seq in d_trees[tree]: # for each string
                elements = seq.split(',')
                for element in elements:
                    sentence = d_body[element]
                    if sentence not in corpus:
                        corpus.append(sentence)

            if not corpus or len(corpus[0]) < 7:
                continue

            # fit: learn vocabulary and idf from training set
            # transform: transform documents to document-term matrix
            tfidv = TfidfVectorizer().fit(corpus)
            tfidv_feature_names = tfidv.get_feature_names()
            tfidv_voca = {}
            for item in tfidv.vocabulary_:
                tfidv_voca[item[0]] = item[1]

            for seq in d_trees[tree]:
                elements = seq.split(',')
                for element in elements:
                    if element not in d:
                        sentence = d_body[element].lower()
                        doc_term_matrix = tfidv.transform([sentence]).toarray().tolist()[0]
                        top_idx_tfidf_lst = zip(*heapq.nlargest(N, enumerate(doc_term_matrix), key=operator.itemgetter(1)))

                        d_topwords[element] = {}
                        d_topwords[element]['words'] = []
                        d_topwords[element]['tfidfs'] = []
                        for j, idx in enumerate(top_idx_tfidf_lst[0]):
                            word = tfidv_feature_names[idx]
                            word_tfidf = top_idx_tfidf_lst[1][j]
                            if word_tfidf == 0:
                                break
                            d_topwords[element]['words'].append(word)
                            d_topwords[element]['tfidfs'].append(word_tfidf)

                        feature_mean = mean_vectorizer.transform(d_topwords[element]['words'])
                        d[element] = {}
                        d[element]['google.mean'] = feature_mean
    except Exception, e:
        print e, i, corpus  

    pickle.dump(d_topwords, open('/home/jhlim/data/top5wordtfidfnostop.p', 'w'))
    pickle.dump(d, open('/home/jhlim/data/top5w2vnostop.p', 'w'))

    sql_close(cursor, conn)

    print 'work time: %s sec'%(time.time()-start_time)
    print '\n\n'
