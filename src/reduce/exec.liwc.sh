#!/bin/bash

for ((i=60; i<93; i++)); do
	echo "liwc.py 1 0 ${i}"
	python3 liwc.py 1 0 ${i}
	echo "liwc.py 2 0 ${i}"
	python3 liwc.py 2 0 ${i}
done

