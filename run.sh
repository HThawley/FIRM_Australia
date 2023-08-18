#!/bin/bash

python3 Optimisation.py -s 21 -z '[7]' -n 25 -v 0
python3 Optimisation.py -s 21 -z '[10]' -n 25 -v 0
python3 Optimisation.py -s 21 -z '[14]' -n 25 -v 0
python3 Optimisation.py -s 21 -z '[3 27 28]' -n 25 -v 0
python3 Optimisation.py -s 21 -z '[18 26]' -n 25 -v 0
python3 Optimisation.py -s 21 -z '[18 28]' -n 25 -v 0
python3 Optimisation.py -s 21 -z '[18 26 28]' -n 25 -v 0
python3 Optimisation.py -s 21 -z '[3 13 27]' -n 25 -v 0

python3 Optimisation.py -s 21 -z all -n 10 -v 0 -i 800
python3 Optimisation.py -s 21 -z all -n 25 -v 0 -i 800

python3 Optimisation.py -s 22 -z '[7]' -n 25 -v 0
python3 Optimisation.py -s 22 -z '[10]' -n 25 -v 0
python3 Optimisation.py -s 22 -z '[14]' -n 25 -v 0
python3 Optimisation.py -s 22 -z '[29]' -n 25 -v 0
python3 Optimisation.py -s 22 -z '[31]' -n 25 -v 0
python3 Optimisation.py -s 22 -z '[3 27 28]' -n 25 -v 0
python3 Optimisation.py -s 22 -z '[18 26]' -n 25 -v 0
python3 Optimisation.py -s 22 -z '[18 28]' -n 25 -v 0
python3 Optimisation.py -s 22 -z '[18 26 28]' -n 25 -v 0
python3 Optimisation.py -s 22 -z '[3 13 27]' -n 25 -v 0
python3 Optimisation.py -s 22 -z all -n 25 -v 0


