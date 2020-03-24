#!/bin/sh

branch=$1

cd DATX02-20-04 && \
git fetch && \
git checkout $branch && \
git pull && \
cp -r ./* ../ && \
cd ..
