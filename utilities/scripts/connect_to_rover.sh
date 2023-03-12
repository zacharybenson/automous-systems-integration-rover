#This script automatically connects to the rover
#!/bin/sh

AFA_GUEST_IP=1
ACCER_IP=1
other_ip=1
PASSVAR=usafa

echo -n "0:AFAGuest, 1:Accer \n"
read wifivar

if [ $wifivar -eq 0 ]
then
  echo "here"
  ssh USAFA@$AFA_GUEST_IP
  $PASSVAR

if [ $wifivar -eq 1 ]
then
  echo "here 1"
  ssh USAFA@$ACCER_IP
  $PASSVAR

fi
