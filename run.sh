#!/bin/bash


python3 Dispatch.py -s 21 -z [7] -n 25 -e e 
python3 Dispatch.py -s 21 -z [10] -n 25 -e e 
python3 Dispatch.py -s 21 -z [14] -n 25 -e e 

python3 Optimisation.py -s 21 -z None -n 25 -e None -x 2 -i 500 -t 1
python3 Optimisation.py -s 21 -z None -n 25 -e None -x 2 -i 500 -t 2
python3 Optimisation.py -s 21 -z None -n 25 -e None -x 2 -i 500 -t 3


