#!/bin/bash

len=100
lastFile=10
mpiexec -f machinefile ./test ./config/configlda${lastFile}.json ./DUnsup/DUnsup_lda_${lastFile}feature.dat

for ((i=15; i<$len+1; i=i+5))
do
	awk 'NR==8{ gsub("${lastFile}","${i}") }{print}' ./config/configlda${i}.json>./config/configlda${i}.json
	mpiexec -f machinefile ./test ./config/configlda${i}.json ./DUnsup/DUnsup_lda_${i}feature.dat
	echo "complete DUnsup_lda_pick{i}feature.dat"
	lastFile=${i}
done