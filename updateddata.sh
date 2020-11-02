#!/bin/bash

cd ./covid-19-data/
git pull

cd ../owid-coviddata/
git pull

cd ../LaDeptHealth
curl -O https://ldh.la.gov/assets/oph/Coronavirus/data/LA_COVID_TESTBYDAY_PARISH_PUBLICUSE.xlsx

cd ../

rm fig*.jpg

python3 CovidData.py

python3 MaskUseOverview.py