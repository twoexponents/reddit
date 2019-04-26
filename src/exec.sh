#!/bin/bash

i=0
while [ $i -lt $1 ]
do
        python sigmoid.py 4 >> len.4.txt
        i=$(($i+1))
done
i=0
while [ $i -lt $1 ]
do
        python sigmoid.py 5 >> len.5.txt
        i=$(($i+1))
done
i=0
while [ $i -lt $1 ]
do
        python sigmoid.py 6 >> len.6.txt
        i=$(($i+1))
done
i=0
while [ $i -lt $1 ]
do
        python sigmoid.py 7 >> len.7.txt
        i=$(($i+1))
done
i=0
while [ $i -lt $1 ]
do
        python sigmoid.py 8 >> len.8.txt
        i=$(($i+1))
done
i=0
while [ $i -lt $1 ]
do
        python sigmoid.py 9 >> len.9.txt
        i=$(($i+1))
done
i=0
while [ $i -lt $1 ]
do
        python sigmoid.py 10 >> len.10.txt
        i=$(($i+1))
done

