import sys
import pickle

def main(argv):
    if len(sys.argv) <= 1:
        print ("sequence len: 1")
        input_length = 1
    else:
        input_length = int(sys.argv[1])

    # 1.1 load feature dataset
    with open('/home/jhlim/reddit/data/contentfeatures.others.p', 'rb') as f:
        d_features = pickle.load(f)

    with open('/home/jhlim/data/commentbodyfeatures.p', 'rb') as f:
        d_commentbodyfeatures = pickle.load(f)

    print ("features are loaded")

    for seq_length in range(input_length, input_length+1):
        '''
        f = open('/home/jhlim/reddit/data/seq.test.%d.csv'%(seq_length), 'r')
        test_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
        f.close()

        f = open('/home/jhlim/data/testbody.tsv', 'w')
        for seq in test_instances:
            try:
                for element in seq[:-1]: # seq[-1] : Y. element: 't3_7dfvv'
                    print (element)
                    if d_features.has_key(element):
                        print ('hi')
                        f.write(element + '\n')
                        for word in d_commentbodyfeatures[element]['body']:
                            f.write(word + ' ')
                        f.write('\n')
            except Exception as e:
                continue
        f.close()
        '''
        f = open('/home/jhlim/data/testbody.tsv', 'w')
        try:
            for element in d_commentbodyfeatures:
                for word in d_commentbodyfeatures[element]['body']:
                    f.write(word + ' ')
                f.write('\n')
        except Exception as e:
            continue
        f.close()

if __name__ == '__main__':
    main([sys.argv])