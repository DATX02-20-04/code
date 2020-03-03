#!/bin/sh

set -e

cd DATX02-20-04
git pull
cp -r src/* ../
cd..
