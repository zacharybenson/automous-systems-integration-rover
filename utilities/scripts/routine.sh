#!/bin/sh

echo sh record.sh
sh record.sh

read -p "Are you ready to process data? " -n 1 -r
# if $REPLY =~ ^[Yy]$
# then
echo sh process.sh
sh process.sh
fi

read -p "Are you ready to transfer data? " -n 1 -r
# if  $REPLY =~ ^[Yy]$
# then
echo sh move_data.sh
sh move_data.sh
# fi