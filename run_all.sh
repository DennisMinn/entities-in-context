#!/bin/bash

sbatch submit.sh "srun python3 train.py -s 0 -e 1062 -f data/runs_1.json"
sbatch submit.sh "srun python3 train.py -s 1062 -e 2124 -f data/runs_2.json"
sbatch submit.sh "srun python3 train.py -s 2124 -e 3186 -f data/runs_3.json"
sbatch submit.sh "srun python3 train.py -s 3186 -e 4250 -f data/runs_4.json"
