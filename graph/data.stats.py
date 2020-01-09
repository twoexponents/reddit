import pymysql
import pickle

def sql_connect():
  conn = pymysql.connect(host="localhost", user="root", passwd="mmlab2019", db="reddit_news")
  cursor = conn.cursor()
  return conn, cursor

def sql_close(cursor, conn):
  cursor.close()
  conn.close()

conn, cursor, = sql_connect()

d_ts = {}; d2_ts={}
d_ict = {}
sql = """
    SELECT post_key, unix_timestamp(created_utc)
    FROM posts
    """
cursor.execute(sql)
rs = cursor.fetchall()

for item in rs:
    pkey, ts, = item
    if pkey not in d2_ts:
        d2_ts[pkey] = ts

sql = """
    SELECT comment_key, unix_timestamp(created_utc), inter_comment_time as ict
    FROM comments
    """
cursor.execute(sql)
rs = cursor.fetchall()

for item in rs:
    ckey, ts, ict, = item
    if ckey not in d_ts:
        d_ts[ckey] = ts
        d_ict[ckey] = ict

f = open('../../data/sequences.csv', 'r')

sum_len = 0
i = 0
min_len = 9999
max_len = 0
min_ts = 99999999999999999
max_ts = 0

lines = f.readlines()
for line in lines:
    sequence = line.replace('\n', '').split(',')
    sum_len += len(sequence)
    if len(sequence) > max_len:
        max_len = len(sequence)
    if len(sequence) < min_len:
        min_len = len(sequence)
    i += 1

print ('i: %d'%(i))
print ('sum_len: %d'%(sum_len))
print ('avg_len: %f'%(sum_len/i))
print ('max_len: %d'%(max_len))
print ('min_len: %d'%(min_len))

cnt = 0
cnt2 = 0
cnt3 = 0
t_ict = 0
ict_lst = []
for line in lines:
    line = line.replace('\n', '').split(',')
    for id in line:
        if id in d_ts:
            if d_ts[id] > 1519862399:
                cnt += 1
                t_ict += d_ict[id]
                ict_lst.append(d_ict[id])
                if d_ts[id] > max_ts:
                    max_ts = d_ts[id]
            else:
                cnt2 += 1

    for id in line:
        if id in d2_ts:
            if d2_ts[id] > 1519862399:
                cnt3 += 1

print ('cnt after 180228: %d, before: %d'%(cnt, cnt2))
print ('max timestamp: %d'%(max_ts))
print ('avg ict: %f'%(t_ict/cnt))
print ('post cnt3: %d'%(cnt3))
print (sorted(ict_lst))

    

