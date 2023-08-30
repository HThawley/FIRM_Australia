#!/bin/bash

python3 Optimisation.py -s 21 -z 'all' -n 25 -v 1 -x 2 -i 20
python3 Optimisation.py -s 22 -z '[3 27 28]' -n 25 -v 0 -x 2 -i 400
python3 Optimisation.py -s 22 -z '[10]' -n 25 -v 0 -x 2 -i 400
python3 Optimisation.py -s 22 -z '[14]'  n 25 -v 0 -x 2 -i 400
python3 Optimisation.py -s 22 -z '[18 26 28]' -n 25 -v 0 -x 2 -i 400
python3 Optimisation.py -s 22 -z '[18 26]' -n 25 -v 0 -x 2 -i 400
python3 Optimisation.py -s 22 -z '[18 28]' -n 25 -v 0 -x 2 -i 400
python3 Optimisation.py -s 22 -z '[7]' -n 25 -v 0 -x 2 -i 400

