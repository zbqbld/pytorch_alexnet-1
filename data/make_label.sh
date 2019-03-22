#!/usr/bin/env sh
DATA=./
MY=./

b=3
echo "Create train.txt..."
rm -rf $MY/train.txt
for i in 3 4 5 6 7 
do
var=`expr $i - $b `
find $DATA/train -name $i*.jpg | cut -d '/' -f4-5 | sed "s/$/ $var/">>$MY/train.txt
done
echo "Create test.txt..."
rm -rf $MY/test.txt
for i in 3 4 5 6 7
do
var=`expr $i - $b `
find $DATA/test -name $i*.jpg | cut -d '/' -f4-5 | sed "s/$/ $var/">>$MY/test.txt
done
echo "All done"
