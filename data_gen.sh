#!/bin/bash
trap "echo 'Script interrupted'; exit" INT

for i in {1..30}
do
  a=$(awk -v min=0 -v max=1 'BEGIN{srand(); print min+rand()*(max-min)}')
  echo "parameter $a" > parameters.txt
  echo "$a" >> parameters_list.txt
  mpirun -np 32 ./gepup_lhs 4
  wait
done