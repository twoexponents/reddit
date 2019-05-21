#!/bin/bash

i=0
while [ $i -lt $1 ]
do
        python feature.user.py 2 >> ../result/user/len2.nosample.txt
        i=$(($i+1))
done
