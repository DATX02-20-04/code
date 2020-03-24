#!/bin/sh

branch=$1

cd DATX02-20-04 && \
git fetch && \
git checkout $branch && \
git pull && \
pip install -r requirements.txt && \
cp -r ./* ../ && \
cd ..
