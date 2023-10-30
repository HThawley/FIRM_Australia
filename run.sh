#!/bin/bash


python3 Optimisation.py -s 21 -z [7] -n 25 -e e -x 3 -i 2500 
python3 Optimisation.py -s 21 -z [10] -n 25 -e e -x 3 -i 2500 
python3 Optimisation.py -s 21 -z [14] -n 25 -e e -x 3 -i 2500 

python3 Optimisation.py -s 21 -z None -n 25 -e None -x 1 -i 500 -t 1
python3 Optimisation.py -s 21 -z None -n 25 -e None -x 1 -i 500 -t 2
python3 Optimisation.py -s 21 -z None -n 25 -e None -x 1 -i 500 -t 3
python3 Optimisation.py -s 21 -z None -n 25 -e None -x 3 -i 2500 -t 4
python3 Optimisation.py -s 21 -z None -n 25 -e None -x 3 -i 2500 -t 5
python3 Optimisation.py -s 21 -z None -n 25 -e None -x 3 -i 2500 -t 6

