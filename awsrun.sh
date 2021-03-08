#!/bin/bash
#this script allows autorun on AWS.  The code should be downloaded using the awsbootstrap.txt script

#install python3 and git
sudo yum install -y python3 git

#download pip3 bootstrap installer
curl -O https://bootstrap.pypa.io/get-pip.py

#run pip3 install
sudo python3 get-pip.py

#install pandas matplotlib and numpy
sudo /usr/local/bin/pip3 install pandas matplotlib numpy 

#clone all submodules
git submodule update --init --recursive --remote --depth 1

#download the La department of health data
mkdir LaDeptHealth
cd LaDeptHealth
curl -O https://ldh.la.gov/assets/oph/Coronavirus/data/LA_COVID_TESTBYDAY_PARISH_PUBLICUSE.xlsx
cd ..

python3 CovidData.py
python3 CovidMaps.py

python3 updatereadme.py

git commit -m "autoupdate figures only from upstream datasources `date`" fig1.jpg fig2.jpg fig3.jpg fig4.jpg fig5.jpg fig6.jpg fig7.jpg fig8.jpg fig9.jpg RecentMap.jpg README.md

git push origin master