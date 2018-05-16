#!/bin/sh
#if [ "$#" -ne 2 ] || ! [ -d "$1" ]; then
#  echo "Usage: $0 DIRECTORY" >&2
#  exit 1
#fi

echo "Running Spark Job"

path=$1
file_name=$2

spark-submit cute_20180225_b33_CSE7322_parta.py $path $file_name

echo "Job ran successfully!"
exit 0
