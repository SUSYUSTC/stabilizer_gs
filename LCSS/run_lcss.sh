#!/bin/bash

#SBATCH --job-name=LCSS_job
#SBATCH --output=LCSS_output.log
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --mem=100G


python test.py
