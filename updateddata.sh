#!/bin/bash

cd ./covid-19-data/
git pull origin master

cd ../owid-coviddata/
git pull origin master

mkdir LaDeptHealth
cd ../LaDeptHealth
curl -O https://ldh.la.gov/assets/oph/Coronavirus/data/LA_COVID_TESTBYDAY_PARISH_PUBLICUSE.xlsx

cd ../

rm fig1.jpg
rm fig2.jpg
rm fig3.jpg
rm fig4.jpg
rm fig5.jpg
rm fig6.jpg
rm fig7.jpg
rm fig8.jpg
rm fig9.jpg

#python3 CovidData.py

#new data is not generated for this dataset
#python3 MaskUseOverview.py

jupyter lab

#update the README.md file
python3 updatereadme.py

git commit -a -m "updated with data from upstream datasources `date`"

echo "DONE!"

