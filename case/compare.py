import sys
import time

def main(argv):
    if len(sys.argv) <= 2:
        print 'Insert two input files to compare.'
        sys.exit()

    f1 = open('%s'%(sys.argv[1]), 'r')
    f2 = open('%s'%(sys.argv[2]), 'r')
    f3 = open('result.txt', 'w')

    f1_instances = map(lambda x:x.replace('\n', ''), f1.readlines())
    f2_instances = map(lambda x:x.replace('\n', ''), f2.readlines())
    
    cnt = 0
    s = set()
    for instance in f1_instances:
        s.add(instance)
    for instance in f2_instances:
        if instance in s:
            cnt += 1
            f3.write('%s\n'%(instance))


    print 'len(%s): %d, len(%s): %d'%(sys.argv[1], len(f1_instances), sys.argv[2], len(f2_instances))
    print '# of same sequence: %d'%(cnt)
    

if __name__ == '__main__':
    main(sys.argv)

            

