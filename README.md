# Updated: February 24 2022  10:42 AM CST

# Covid-19 Louisiana Data Compared

**This project is NOT a scientific study!!!  Conclusions should not be drawn from the information contained here.  This is mostly an exercise in data processing and visualization.  For information you can draw conclusion on, please visit the [CDC's Covid19 page](https://www.cdc.gov/coronavirus/2019-ncov/index.html)**

**This project is a replacement for the garyscorner for of nytimes/covid-19-data which will no longer be updated**

[Watch the Maps video on Youtube](https://youtu.be/RiEHIBp87I8)

[Watch the Mask video on Youtube](https://youtu.be/4GHW_iREiJE)

[View Main Project](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/CovidData.ipynb)

[View Maps and Maps Video Project](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/CovidMaps.ipynb)

[View Mask Use Graph Project](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/MaskUseOverview.ipynb)

[View Video Project](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/MaskUseRatesVideo.ipynb)

## Description
This is a simple project aimed at visualizing and comparing Covid-19 data from Louisiana, a state which is very important to us (people from Louisiana).  The main analysis/visualization is in the file CovidData.py, which pulls data from one or more submodules or locations.  Other python notebooks (MaskUseRates.ipynb/MaskUseOverview.ipynb) contained in this project are not updated daily.

## Datasets

* **[Louisiana Department of Health](https://ldh.la.gov/Coronavirus/)** - Data on new cases by parish as well as testing numbers.  This data comes out slower then the other data sources (every two weeksish) and so graphs using parish data, are behind graphics using NYTimes/OWID data.  Please look at the dates on the graphics carefully before making any comparisons between different data sets.
* **[nytimes/covid-19-data](https://github.com/nytimes/covid-19-data)** - Data on new cases and new deaths in the US and Louisiana were taken from this dataset.  Nytimes data was used in most places for Louisiana, but in some places an reduction of the LaDH data was used to provide better comparison with parish data.
* **[Our World in Data](https://github.com/owid/covid-19-data/)** - World Covid-19 data and [world population](https://ourworldindata.org/world-population-growth)
* **[U.S. Census Bureau](https://www.census.gov)** - Population of the US and of Louisiana were taken from the Census Bureau and used for per capita comparisons.  Additional data on [FIPS codes](https://www.census.gov/2010census/xls/fips_codes_website.xls) by parish where taken from U.S. Census Bureau.
* **[U.S. Geological Survey](https://pubs.usgs.gov/of/1998/of98-805/html/gismeta.htm)** - Shape files for inlane water.

Datasets taken from LaDH are dated by sample collection date, but NyTimes and OWID data is by reported date, so there is an offset in the dates between New Orleans/EBR and other datasets.  Please make note of this when drawing any conclusions.

## Maps

[Watch the Maps video on Youtube](https://youtu.be/RiEHIBp87I8)

![Recent Map](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/RecentMap.jpg)

## Output

![Figure 1](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/fig1.jpg)
![Figure 2](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/fig2.jpg)
![Figure 3](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/fig3.jpg)
![Figure 4](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/fig4.jpg)
![Figure 8](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/fig8.jpg)
![Figure 5](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/fig5.jpg)
![Figure 6](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/fig6.jpg)
![Figure 7](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/fig7.jpg)
![Figure 9](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/fig9.jpg)
![Figure 10](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/fig10.jpg)

## Video

If you are drawing conclusions other then, "Wow, that's a neat graph", then you have made a mistake.  One should make note that there are a lot of factors that are correlated to mask use, population size, and infection rates, and just because two things are correlated does not mean they are causative.  For example in just this video you can see that there is a correlation between mask use and population size, a passing glance at the labels will tell us that the population densities in areas where mask use it common are probably higher.  Population density is one of those things you would expect to have influence on infection rates.  So again, if you are drawing conclusions from this video, know that you shouldn't. The author of this work makes no clames about the effectives of wearing a mask, population size, or anything else on infection rates.  If you want data and visualization upon which you SHOULD draw conclusion, please visit the [CDC's Covid19 page](https://www.cdc.gov/coronavirus/2019-ncov/index.html)

[Watch the video on Youtube](https://youtu.be/4GHW_iREiJE)

The below is an automatically generated screenshot, and is a single frame from the video linked above.  It doesn't mean much when viewed alone.
![Screen_Shot](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/MaskUseVidScrShot.jpg)

## Files

* **.gitignore** - lists file patterns to be ignored by git
* **.gitmodules** - provides linking to submodules
* **CovidData.ipynb** - Jupyter notebook file, generates figures, and provides daily analysis
* **CovidMaps.ipynb** - Jupyter notebook file, generates maps, and map video.
* **LICENSE** - GNU General Public License v3.0
* **MaskUseOverview.ipynb** - Jupyter notebook file, generates the mask use rages bar graph for Louisiana parishes
* **MaskUseRatesVideo.ipynb** - Jupyter notebook file, generates maskuse/new cases rate comparision video (link at top)
* **MaskUseVidScrShot.jpg** - latest generated screenshot from MaskUseRagesVideo.ipynb
* **README.md** - readme file serves as projects main page, providing all nessisary information about the project and it's analysis, this is automatically generated from the README.template.md whenever the project is auto-updated
* **README.template.md** - template for the readme file which is used to generate an updated README.md when the project is auto-updated
* **fig1-10.jpg** - Figure images (JPEG) for the project are generated by the project notebook files and are the main output of this project.
* **getcensusdata.sh** - bash script which downloads the census data required for this project, and prevents having to add the data as part of this git *(run once)*
* **getUSGSdata.sh** - bash script which downloads inland waterway shapefiles from USGS
* **updateddata.sh** - bash script which updates the project automagically with the most recent covid data
* **updatereadme.py** - python script which updates the README.md file using the README.template.md file to include the current date/time

## Limitation

This is not a scientific study please don't treat it like one.  This is a data visualization and processing project.  Please keep that in mind.

The analysis here is done with data from outside sources such as the [Louisiana Department of Health](https://ldh.la.gov/) and the [NYTimes](https://github.com/nytimes/covid-19-data/).  The sources have no affiliation with this project, and the projects maintainer makes no claims about the accuracy of the sources.  Essentially if you have an issue with what you see in the data, you should probably take it up with the upsteam data souces.

Please excuse spelling, I hammered this out pretty quick, and if you are reading this it means I didn't follow through on circling back to fix it :(

## Liabilities
This project should be interesting and nothing more, it is not intended for research purposed or policy decisions.  No decisions, assumptions, accusations, etc stemming from the analysis here should be made, and anyone who does so will do so at their own risk.  Any liability resulting from the use of this analysis is solely the end user's responsibility. 

