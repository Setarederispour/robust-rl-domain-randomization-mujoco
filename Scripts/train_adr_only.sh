#!/bin/bash
python train.py --domain source --env_family hopper --adr
python train.py --domain source --env_family walker2d --adr
python train.py --domain source --env_family walker2d --adr --total_timesteps 3000000
