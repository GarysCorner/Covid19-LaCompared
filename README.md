# Covid-19 Louisiana Data Compared

**This project is a replacement for the garyscorner for of nytimes/covid-19-data which will no longer be updated**

[Watch the video on Youtube](https://youtu.be/pAnK_I578PA)

[View Main Project](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/CovidData.ipynb)
[View Mask Use Graph Project](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/MaskUseOverview.ipynb)
[View Video Project](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/MaskUseRatesVideo.ipynb)

## Description
This is a simple project aimed at visualizing and comparing Covid-19 data from Louisiana, a state which is very important to us (people from Louisiana).  The main analysis/visualization is in the file CovidData.py, which pulls data from one or more submodules or locations.

## Datasets

* **[Louisiana Department of Health](https://ldh.la.gov/Coronavirus/)** - Data on new cases by parish as well as testing numbers.  This data comes out slower then the other data sources (every two weeksish) and so graphs using parish data, are behind graphics using NYTimes/OWID data.  Please look at the dates on the graphics carefully before making any comparisons between different data sets.
* **[nytimes/covid-19-data](https://github.com/nytimes/covid-19-data)** - Data on new cases and new deaths in the US and Louisiana were taken from this dataset.  Nytimes data was used in most places for Louisiana, but in some places an reduction of the LaDH data was used to provide better comparison with parish data.
* **[Our World in Data](https://github.com/owid/covid-19-data/)** - World Covid-19 data and [world population](https://ourworldindata.org/world-population-growth)
* **[U.S. Census Bureau](https://www.census.gov)** - Population of the US and of Louisiana were taken from the Census Bureau and used for per capita comparisons.  Additional data on [FIPS codes](https://www.census.gov/2010census/xls/fips_codes_website.xls) by parish where taken from U.S. Census Bureau.

Datasets taken from LaDH are dated by sample collection date, but NyTimes and OWID data is by reported date, so there is an offset in the dates between New Orleans/EBR and other datasets.  Please make note of this when drawing any conclusions.

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

[Watch the video on Youtube](https://youtu.be/pAnK_I578PA)

![Screen_Shot](https://github.com/GarysCorner/Covid19-LaCompared/blob/master/MaskUseVidScrShot.jpg)


## Limitation
The analysis here is done with data from outside sources, which themselves are compiling data from other sources, which themselves...  Some of these sources have changed for the US, and there are reasons to believe that some of them may have issues with reliability, and so the analysis here can only be as good as the underlying data, and this author makes no claims about that data.

Please excuse spelling, I hammered this out pretty quick, and if you are reading this it means I didn't follow through on circling back to fix it :(

## Liabilities
This project should be interesting and nothing more, it is not intended for research purposed or policy decisions.  No decisions, assumptions, accusations, etc stemming from the analysis here should be made, and anyone who does so will do so at their own risk.  Any liability resulting from the use of this analysis is solely the end user's responsibility. 

### Liabilities TLDR
It's all on you bro, don't use this and then come blame me.

