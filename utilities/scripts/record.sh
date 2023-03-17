#! /bin/bash

echo cd /home/github/automous-systems-integration-rover
cd /home/github/automous-systems-integration-rover

echo python3 rover_recorder.py
python3 rover_recorder.py

echo python3 rover_data_processor.py
python3 rover_data_processor.py
echo 'Data Recording & Processing Complete'