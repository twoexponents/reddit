from operator import itemgetter
from collections import Counter

f = open('result.txt', 'r')

lines = f.readlines()

d_words = {}; d_values = {}
for line in lines:
    items = line.replace('\n', '').split('\t')
    body = items[1]

    words = body.split(' ')
    for word in words:
        if word not in d_words:
            d_words[word] = {}
            d_words[word] = 1
            d_values[word] = items[2]
        else:
            d_words[word] += 1
            d_values[word] += (' ' + items[2])

items = d_words.items()
sorted_items = list(sorted(items, key=itemgetter(1)))
print (sorted_items[:10])
#reversed_items = reversed(items)

sorted_items.reverse()

for item in sorted_items[:500]:
    c = Counter(d_values[item[0]].split(' '))
    most_common = c.most_common()
    first = most_common[0]
    second = None
    if len(most_common) > 1:
        second = most_common[1]
    print (item, first, second, first[1]/item[1])

#print (sorted_items[-50:])

#print (reversed_items[:10])
