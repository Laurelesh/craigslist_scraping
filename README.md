# Too good to be true

## Use Cases:

This repo includes an algorithm to detect anomalously inexpensive housing listings on craigslist.

The most obvious application is to flag craigslist scams. In this iteration, since price per room for the given neighborhood is the only input feature, the algorithm does not distinguish between a scam and a "deal".  Future iterations could separate the flagged listings into "probably scam" vs "probably deal", most simply by defining a range of prices that are good but not _too_ good ("deal"), or more elaborately by cross-checking listing descriptions with those on real estate websites like zillow (from which scammers often copy-paste images and descriptions, then post on craigslist as if for rent).

## Methods:

Using BeautifulSoup, I scrape housing data from craigslist.

I use Pandas in a Jupyter notebook to clean and standardize the data, and to write a simple anomaly detector assuming a log-normal distribution for price per bedroom.

## To Run:

Open and run the jupyter notebook `craigslist_scraping_v2.ipynb`.

### Dependencies:

* python 3
* Jupyter


