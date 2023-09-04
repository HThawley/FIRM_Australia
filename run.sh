#!/bin/bash

python3 Optimisation.py -s 21 -z 'all'          -n 25 -v 0 -x 2 -i 2000
python3 Optimisation.py -s 21 -z '[10]'         -n 25 -v 0 -x 1 -i 2500
python3 Optimisation.py -s 21 -z '[14]'         -n 25 -v 0 -x 1 -i 2500
python3 Optimisation.py -s 21 -z '[3 27 28]'    -n 25 -v 0 -x 1 -i 2500
python3 Optimisation.py -s 21 -z '[18 26]'      -n 25 -v 0 -x 1 -i 2500
python3 Optimisation.py -s 21 -z '[18 28]'      -n 25 -v 0 -x 1 -i 2500
python3 Optimisation.py -s 21 -z '[18 26 28]'   -n 25 -v 0 -x 1 -i 2500

