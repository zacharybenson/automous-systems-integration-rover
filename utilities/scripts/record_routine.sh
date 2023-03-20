#!/bin/shell
echo cd /home/github/automous-systems-integration-rover
cd /home/github/automous-systems-integration-rover

echo python3 rover_recorder.py
python3 rover_recorder.py

echo python3 rover_data_processor.py
python3 rover_data_processor.py
echo 'Data Recording & Processing Complete'

read -p "Are you ready to process data? " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
	echo cd /home/github/automous-systems-integration-rover
	cd /home/github/automous-systems-integration-rover

	echo python3 rover_data_processor.py
	python3 rover_data_processor.py
	echo 'Processing Complete'
fi

read -p "Are you ready to transfer data? " -n 1 -r
if  [[ $REPLY =~ ^[Yy]$ ]]
then
	# move data from data file to hard drive
	echo mv /home/usafa/data/* #put hardrive here
	mv /home/usafa/data/* #put hardrive here
fi