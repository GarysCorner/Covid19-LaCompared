#!/bin/bash

cd ./covid-19-data/
git pull

cd ../owid-coviddata/
git pull

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

git commit -m "updated with data from upstead datasources `date`" fig1.jpg fig2.jpg fig3.jpg fig4.jpg fig5.jpg fig6.jpg fig7.jpg fig8.jpg fig9.jpg CovidData.ipynb

echo "DONE!"

