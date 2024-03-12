#!/bin/bash

sudo apt-get update -y

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && bash ~/miniconda.sh -b -p $HOME/miniconda
echo 'eval "$(~/miniconda/bin/conda shell.bash hook)"' >> ~/.bashrc
source ~/.bashrc
