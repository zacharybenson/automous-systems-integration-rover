#!/bin/shell
#! /bin/bash
echo ssh usafa@10.1.100.236
ssh usafa@10.1.100.236 <<ENDSSH
echo cd /home/github/automous-systems-integration-rover/data_collection
cd /home/github/automous-systems-integration-rover/data_collection

echo python3 rover_recorder.py
python3 rover_recorder.py

echo python3 rover_data_processor.py
python3 rover_data_processor.py
echo 'Data Recording & Processing Complete'

read -p "Are you ready to process data? " -n 1 -r

echo python3 rover_data_processor.py
python3 rover_data_processor.py
echo 'Processing Complete'

read -p "Are you ready to transfer data? " -n 1 -r
# move data from data file to hard drive
echo mv /home/usafa/data/* /home/usafa/extern_data/Benson
mv /home/usafa/data/* /home/usafa/extern_data/Benson

ENDSSH