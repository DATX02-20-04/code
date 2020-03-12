#!/usr/bin/sh

set -e

for FILE in src/*.test.py;
do
    python $FILE
done
