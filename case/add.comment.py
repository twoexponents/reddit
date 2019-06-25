import sys
import time

def main(argv):
    if len(sys.argv) <= 2:
        print 'Insert two input files to compare.'
        sys.exit()

    f1 = open('%s'%(sys.argv[1]), 'r')
    f2 = open('%s'%(sys.argv[2]), 'r')
    f3 = open('output.txt', 'w')

    f1_instances = map(lambda x:x.replace('\n', ''), f1.readlines())
    f2_instances = map(lambda x:x.replace('\n', ''), f2.readlines())
    
    cnt = 0
    s = set()
    for instance in f1_instances:
        s.add(instance)

    for idx in range(len(f2_instances)):
        if f2_instances[idx] in s:
            f3.write('%s\n'%(f2_instances[idx]))
            f3.write('%s\n'%(f2_instances[idx+1]))

if __name__ == '__main__':
    main(sys.argv)

            

