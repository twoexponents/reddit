f = open('liwc.out3.txt')

lines = f.readlines()



d = {}
max_auc = 0; element = ''; length = ''
for line in lines:
    words = line.replace('\n', '').split()
    if len(words) == 0:
        continue
    if words[0] == 'liwc.py' and words[1] == '1':
        max_auc = 0
        element = words[3]
        length = words[1]
    if words[0] == 'seq_length:' and words[1] == '1,' and words[-2] == 'auc:':
        auc = float(words[-1])
        if auc > max_auc:
            max_auc = auc
            d[element] = {}
            d[element][length] = max_auc
 
d2 = {}
for line in lines:
    words = line.replace('\n', '').split()
    if len(words) == 0:
        continue
    if words[0] == 'liwc.py' and words[1] == '2':
        max_auc = 0
        element = words[3]
        length = words[1]
    if words[0] == 'seq_length:' and words[1] == '2,' and words[-2] == 'auc:':
        auc = float(words[-1])
        if auc > max_auc:
            max_auc = auc
            d2[element] = {}
            d2[element][length] = max_auc

for i in range(len(d2)):
    idx = i+60
    print (idx, d[str(idx)]['1'], d2[str(idx)]['2'])
        

        


