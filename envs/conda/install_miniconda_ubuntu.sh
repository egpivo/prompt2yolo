#!/bin/bash

# Update package lists
sudo apt update

# Download Miniconda installer
sudo wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda-installer.sh

# Install Miniconda silently
bash /opt/miniconda-installer.sh -b -p $HOME/miniconda

echo 'source $HOME/miniconda/bin/activate' >> ~/.bashrc

# Update shell environment
source ~/.bashrc
