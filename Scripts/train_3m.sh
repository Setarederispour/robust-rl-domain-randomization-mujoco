#!/bin/bash
python train.py --domain source --env_family walker2d --no-udr  --total_timesteps 3000000
python train.py --domain source --env_family walker2d --udr --total_timesteps 3000000
python train.py --domain source --env_family walker2d --adr --total_timesteps 3000000
