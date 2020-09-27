#!/bin/bash

cd ./covid-19-data/
git pull

cd ../owid-coviddata/
git pull


cd ../

jupyter notebook
