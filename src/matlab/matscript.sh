#!/bin/bash

start=1
end=11

for((i=$start;i<=$end;i++))
do
    chmod +x "mat${i}.sh"
    sbatch "mat${i}.sh"
    sleep 120
    # if [ $(($i % 3)) -eq 0 ];
    # then
    #     sleep 120
    # else
    #     sleep 20
    # fi
done

