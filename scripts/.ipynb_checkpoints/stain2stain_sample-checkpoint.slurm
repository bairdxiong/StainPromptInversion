#!/bin/bash
#SBATCH -J hdtesttain_sample_test
#SBATCH -p ksagnormal01
#SBATCH -N 1 
#SBATCH -n 32
#SBATCH --gres=gpu:1


bash scripts/stain2stain_sample.sh
