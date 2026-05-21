#!/bin/bash

python train.py --domain source --env_family walker2d --no-udr
python train.py --domain source --env_family walker2d --udr
python train.py --domain source --env_family walker2d --adr

python train.py --domain source --env_family hopper --no-udr
python train.py --domain source --env_family hopper --udr
python train.py --domain source --env_family hopper --adr
