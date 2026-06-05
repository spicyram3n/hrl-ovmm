#!/bin/bash
sudo chown -R $(whoami) /home/ws
echo 'export PYTHONPATH=$PYTHONPATH:/home/ws/source' >> ~/.bashrc
