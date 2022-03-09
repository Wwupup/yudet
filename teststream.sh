#!/usr/bin/bash

target="facenvive"
prefix="/home/ww/projects/yudet/workspace/${target}/weights/"
python test.py -m ${prefix}${target}_final.pth

begin=1000
end=500
while ((begin>end))
do
    python test.py -m ${prefix}${target}_epoch_$begin.pth
    ((begin-=50))
done
