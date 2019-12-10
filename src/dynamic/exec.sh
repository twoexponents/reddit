#input_length, hidden_size, learning_rate, batch_size, keep_rate, seed1, seed2
#1) change hidden_size
:<<'END'
echo "1) change hidden_size"
python3 dynamic.bert.liwc.cont.time.py 3 32 0.001 128 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 64 0.001 128 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 128 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 150 0.001 128 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 200 0.001 128 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 256 0.001 128 0.5 10 10

echo "2) change learning_rate"
python3 dynamic.bert.liwc.cont.time.py 3 128 0.0005 128 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 128 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 128 0.002 128 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 128 0.005 128 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 128 0.01 128 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 128 0.02 128 0.5 10 10

echo "3) change batch_size"
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 32 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 64 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 128 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 200 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 256 0.5 10 10

echo "4) change keep_rate"
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 128 0.5 10 10
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 128 0.6 10 10
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 128 0.7 10 10
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 128 0.8 10 10
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 128 0.9 10 10
END

echo "5) change seed"
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 128 0.9 300 50
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 128 0.9 300 100
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 128 0.9 300 200
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 128 0.9 300 300
python3 dynamic.bert.liwc.cont.time.py 3 128 0.001 128 0.9 300 400


