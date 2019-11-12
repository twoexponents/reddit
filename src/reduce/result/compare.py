import sys
import time
import pickle


# find cases that out1 predicts fail but out2 predicts success
def main(argv):
    if len(sys.argv) <= 2:
        print ('Insert two input files to compare.')
        sys.exit()

    sentencefile = pickle.load(open('/home/jhlim/data/commentbodyfeatures.p', 'rb'))

    f1 = open('%s'%(sys.argv[1]), 'r')
    f2 = open('%s'%(sys.argv[2]), 'r')
    f3 = open('result.txt', 'w')

    f1_instances = list(map(lambda x:x.replace('\n', ''), f1.readlines()))
    f2_instances = list(map(lambda x:x.replace('\n', ''), f2.readlines()))
    
    cnt = 0
    s = set()
    for instance in f1_instances:
        items = instance.replace('\n', '').split('\t')
        s.add(items[0])
    for instance in f2_instances:
        items = instance.replace('\n', '').split('\t')

        if items[0] not in s:
            cnt += 1
            body = 'NULL'
            if items[0] in sentencefile:
                body = sentencefile[items[0]]
            f3.write(items[0] + '\t' + body + '\t' + items[1] + '\t' + items[2] + '\n')

    print ('len(%s): %d, len(%s): %d'%(sys.argv[1], len(f1_instances), sys.argv[2], len(f2_instances)))
    print ('# elements not in file1: %d'%(cnt))
    

if __name__ == '__main__':
    main(sys.argv)

            

