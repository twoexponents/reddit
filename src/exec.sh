#!/bin/bash

i=0
while [ $i -lt $1 ]
do
        python feature.all.py 3 
        i=$(($i+1))
done
