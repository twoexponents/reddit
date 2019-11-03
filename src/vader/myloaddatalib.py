import pickle

def load_userfeatures():
    return pickle.load(open('/home/jhlim/data/userfeatures.activity.p', 'rb'))

def load_bertfeatures(input_length=1):
    return pickle.load(open('/home/jhlim/data/bertfeatures' + str(input_length) + '.p', 'rb'))

def load_contfeatures():
    return pickle.load(open('/home/jhlim/data/contentfeatures.others.p', 'rb'))

def load_bodyfeatures():
    return pickle.load(open('/home/jhlim/data/commentbodyfeatures.p', 'rb'))


