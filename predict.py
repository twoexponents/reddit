import tensorflow as tf
import numpy as np
import sys
import cPickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter
from operator import itemgetter

common_features_fields = ['vader_score', 'vader', 'difficulty']
post_features_fields = ['pub_1h', 'pub_hd', 'pub_1d', 'max_similarity_1h',
	'max_similarity_hd', 'max_similarity_1d', 'pub_time_0', 'pub_time_1',
	'pub_time_2', 'pub_time_3', 'pub_time_4', 'pub_time_5', 'pub_time_6',
	'pub_time_7', 'pub_time_8', 'pub_time_9'
