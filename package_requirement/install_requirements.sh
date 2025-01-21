#!/bin/bash

# Check Python version
python --version

sudo apt-get update -y
sudo apt-get install -y build-essential libomp-dev


# Downgrade pip to version 20.2.4
python -m pip install --upgrade pip==23.2

# Upgrade Azure CLI
pip install azure-cli==2.67.0

# Upgrade Azure ML SDK
pip install azureml-sdk==1.59.0

# Install dependencies from requirements.txt
pip install -r requirements.txt

conda update -n base -c defaults conda -y
