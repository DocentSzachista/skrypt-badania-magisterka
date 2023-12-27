#!/bin/bash
python3 -m venv virtual_env
source virtual_env/bin/activate
pip3 install -r requirements.txt
python3 main.py
