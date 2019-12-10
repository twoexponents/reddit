f1 = open("rnn.out.txt", 'r')
f2 = open("leaf.out.txt", 'r')

lines = f2.readlines()
f2.close()
d_leaf = {}
for line in lines:
    elements = line.replace('\n', '').split('\t')
    id = elements[0]; pred = elements[1]; label = elements[2]
    d_leaf[id] = (pred, label)

lines = f1.readlines()
d_rnn = {}
d_only_rnn = {}
for line in lines:
    elements = line.replace('\n', '').split('\t')
    id = elements[0]; pred = elements[1]; label = elements[2]
    d_rnn[id] = (pred, label)
    if id not in d_leaf.keys():
        d_only_rnn[id] = (pred, label)

d_only_leaf = {}
for item in d_leaf.keys():
    if item not in d_rnn.keys():
        d_only_leaf[item] = d_leaf[item]

for item in d_only_rnn.items():
    print (item)
print ('leaf item size: ', len(d_leaf))
print ('rnn item size: ', len(d_rnn))
print ('only in rnn size: ', len(d_only_rnn))
print ('only in leaf size: ', len(d_only_leaf))

