#!/bin/bash

echo "This should only be run once"

rm -R ./uscensus
mkdir uscensus
cd uscensus

curl -O https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv

curl -O https://www2.census.gov/programs-surveys/popest/geographies/2018/all-geocodes-v2018.xlsx

curl -O https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_500k.zip
unzip cb_2018_us_county_500k.zip

curl -O https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv

cd ..

echo "US Census data downloaded"