f1 = open('predicts.txt', 'r')
f2 = open('labels.txt', 'r')

cnt = 0
f1_lst = f1.readlines()
f2_lst = f2.readlines()
size = len(f1_lst)
for l1, l2 in zip(f1_lst, f2_lst):
    if l1 == l2:
        cnt += 1

print (cnt, '/', size)

f1.close()
f2.close()

