#!/bin/bash

i=0
while [ $i -lt $1 ]
do
        python feature.content.py 6
        i=$(($i+1))
done
