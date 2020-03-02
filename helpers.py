"""
Helper functions for scraping, cleaning, and analyzing craigslist housing listings.
"""
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from requests import get
import time
from urllib.request import urlopen


# Scraping

def scrape_housing_data(url = "https://sfbay.craigslist.org/search/apa?availabilityMode=0", studiobedrooms = 1.0):

    """
    Args:
        url: url of the webpage to scrape. Defaults to bay area craigslist.

    Returns:
        dataframe with (almost) raw scraped data (with minor cleaning)

    """

    html = urlopen(url)
    soup = BeautifulSoup(html, 'lxml')

    # define ranges
    rangefrom=int(soup.find(class_="rangeFrom").text)  # starting index for listings
    rangeto=int(soup.find(class_="rangeTo").text)  # ending index for listings
    perpage = rangeto-rangefrom+1  # listings per page
    totalcount = int(soup.find(class_="totalcount").text)  # total number of listings

    #initialize lists of info:
    links=[]  # url links for each listing
    descs=[]  # description for each listing
    prices=[]  # price for each listing
    sqfeets=[]  # square feet for each listing
    bedrooms = []  # number of bedrooms for each listing
    hood=[]  # neighborhood for each listing

    #for each page, make a soup, and for each listing in that soup, append to each info list.
    for i in range(0,totalcount,perpage):
        url = "https://sfbay.craigslist.org/search/apa?availabilityMode=0&s="+str(i)

        response = get(url)
        soup = BeautifulSoup(response.text, 'lxml')#, features = 'lxml')
        resultrows = soup.find_all(class_="result-row");

        ###
        for row in resultrows:

            #hyperlink:
            linki = row.find(class_='result-title hdrlnk')['href']
            links.append(linki)

            #description (or title) of listing:
            description = row.find(class_='result-title').text
            descs.append(description)

            #price:
            try:
                price=float(  re.compile('[\d]+').search( row.find(class_='result-price').text   ).group()   )
                prices.append(price)
            except:
                prices.append(np.NaN)

            #number of bedrooms and square footage:
            try:
                housinginfo = row.find(class_='housing').text
                #bedrooms:
                try: #check whether # of bedrooms is given in the "housing" class:
                    br = float(re.compile('\d').search(  re.compile('\dbr').search(housinginfo).group()  ).group()  ) #this mess pulls out number of bedrooms
                except AttributeError: #if not, try to find bedroom number in description:
                    try:
                        br=re.compile('\d').search(re.compile('(\dbr|\d-br|\d\sbr|\dbdr|\d-bdr|\d\sbedroom|\d-bedroom)').search(description.lower()).group()).group()

                    except: #or maybe the description says it's a studio:
                        stud=re.compile('studio').search(description.lower())
                        if stud is not None:
                            br = studiobedrooms
                        else:
                            br=np.NaN
                #square footage:
                try: #check whether square footage is given in "housing" class:
                    sqfeet = float(re.compile('[\d]+').search( re.compile('[\d]+ft2').search(housinginfo).group() ).group()   )
                    sqfeets.append(sqfeet)
                except AttributeError: #if not, mark "nan" and move on.
                    sqfeets.append(np.NaN)
            except AttributeError: #in case 'housing' class doesn't exist for the listing:
                sqfeets.append(np.NaN) #give up on square footage
                #
                try: #try to find number of bedrooms in description:
                    br=re.compile('\d').search(re.compile('(\dbr|\d-br|\d\sbr|\dbdr|\d-bdr|\d\sbedroom|\d-bedroom)').search(description.lower()).group()).group()
                except: #if no appearance of "br" in description, look for "studio":
                    stud=re.compile('studio').search(description.lower())
                    if stud is not None:
                        br = studiobedrooms
                    else:
                        br=np.NaN
            finally: #br has now been defined, even if as "nan"
                bedrooms.append(float(br))


            #neighborhood
            try:
                nh = re.compile('[^)^(]+').search(  row.find(class_='result-hood').text.strip().lower()  ).group()
                hood.append(nh)
            except:
                hood.append(np.NaN)



        time.sleep(3) #take a break before scraping the next page

    raw_scraped_data = pd.DataFrame({'url': links, 'description': descs, 'price': prices, 'square_feet': sqfeets, 'bedrooms': bedrooms, 'neighborhood': hood})
    return raw_scraped_data

# Cleaning

def get_cleaned_neighborhoods(scraped_data):

    """
    Produce a list from scraped_data['neighborhood'] with no null values, no street addresses, and no slashes, e.g. separate "richmond/antioch" to "richmond", "antioch"

    Args:
        scraped_data: dataframe with a column 'neighborhood' containing strings and np.nan

    Returns:
        list of strings

    """

    raw_nh_list = [nh for nh in list(set(scraped_data['neighborhood'].dropna())) if re.compile('[\d]').search(nh) is None]

    nested_nh_list = [nh.split('/') for nh in raw_nh_list]  # split on '/'
    ## also split on ',' ?
    nh_list = [x.strip().lower() for nh in nested_nh_list for x in nh]  # flatten the list
    nh_list = [x for x in nh_list if re.compile('[\d]').search(x) is None]  # lose the elements containing digits
    nh_list = [x for x in nh_list if x != 'downtown'] # remove 'downtown'
    return nh_list



def extract_neighborhood(description, neighborhoods_list):

    """
    Search for any of the elements of neighborhoods_list in description.

    Args:
        description: text string
        neighborhoods_list: list of strings

    Returns:
        the first element of neighborhoods_list that appears in desctription.

    """

    description = description.lower()
    neighborhoods_list = [x.lower() for x in neighborhoods_list]

    orhoods = r'(\b'+(r'\b|\b'.join(neighborhoods_list))+r'\b)'

    nh_in_description = re.compile(orhoods).search(description)

    if nh_in_description is not None:
        nh = nh_in_description.group()
#         print(f"found {nh} in description {description}")
    else:
        nh = np.nan

    return nh


def fill_neighborhood_from_description(scraped_data):

    """
    Operates on the full dataframe.
    If missing data in 'neighborhood' column of scraped_data, check for 'neighborhood' from any row in 'description' column.

    Args:
        scraped_data: dataframe, must have columns ['description', 'neighborhood']

    Returns:
        copy of scraped_data with missing 'neighborhood' values filled using description, when possible.

    functional dependencies:
        get_cleaned_neighborhoods
        extract_neighborhood
    """

    df = scraped_data.copy()

    neighborhoods_list = get_cleaned_neighborhoods(scraped_data)
#     neighborhoods_list = list(set(scraped_data['neighborhood'].dropna()))


    df['neighborhood'] = df.apply(lambda row: extract_neighborhood(row['description'], neighborhoods_list) if str(row['neighborhood'])=='nan' else row['neighborhood'], axis = 1)


    return df

def street_address_to_neighborhood(possible_street_address, neighborhoods_list):
    """
    If possible_street_address is not contained in neighborhoods_list, return whichever part of it is (if any).
    Args:
        possble_street_address: string, possibly containing digits
        neighborhoods_list
    Returns:
        string

    """
#     possible_street_address = possible_street_address.lower()
    neighborhoods_list = [x.lower() for x in neighborhoods_list]

    orhoods = r'(\b'+(r'\b|\b'.join(neighborhoods_list))+r'\b)'


    if (possible_street_address not in neighborhoods_list) and (str(possible_street_address) != 'nan'):

        neighborhood_within_address = re.compile(orhoods).findall(possible_street_address.lower())
        if neighborhood_within_address != []:

            return('/'.join(neighborhood_within_address))

        else:
            return np.nan

    else:
        return possible_street_address

def convert_street_addresses(scraped_data):

    """
    Operates on the full dataframe.

    If scraped_data['neighborhood'] is not in the cleaned list of neighborhoods (e.g. because it has digits in it),
    check whether it contains any of the cleaned neighborhoods. If so, replace with that. If not, replace with np.nan.

    Args:
        scraped_data: dataframe with column ['neighborhood']

    Returns:
        copy of scraped_data with 'neighborhood' replaced by np.nan if it contained digits

    """
    clean_nh_list = get_cleaned_neighborhoods(scraped_data)

    df = scraped_data.copy()

    df['neighborhood'] = df['neighborhood'].apply(lambda nh: street_address_to_neighborhood(nh, clean_nh_list))

    return df

def scrape_neighborhoods_from_wikipedia(url):

    """
    Args:
        url: url for wikipedia's list of neighborhoods in the desired city
    Returns:
        list of neighborhoods in the given city
    """

    # url = "https://en.wikipedia.org/wiki/List_of_neighborhoods_in_" + '_'.join([x.capitalize() for x in city.split(' ')])

    html = urlopen(url)

    soup = BeautifulSoup(html,'lxml')

    allsections=soup.select('.toclevel-1')

    hoods=[];
    for sec in allsections:
        hoods.append(re.compile("[a-z\s/'-]+").search(sec.text.lower()).group().strip())

    hoods = [nh for nh in hoods if nh.lower() not in ['see also','references', 'external links']]

    return hoods



def lump_neighborhoods_function(nh, sfhoods, oakhoods):

    """
    Args:
        nh: string or np.nan representing the name of a neighborhood.
        sfhoods: list of san francisco neighborhoods
        oakhoods: list of oakland neighborhoods
    Returns: 'san francisco' if nh is in sfhoods, 'oakland' if nh is in oakhoods, or 'san jose' if nh contains 'san jose'

    functional dependencies: scrape_neighborhoods_from_wikipedia (used to define sfhoods and oakhoods).

    """

    if nh in sfhoods:
        new_nh = 'san francisco'

    elif nh in oakhoods:
        new_nh = 'oakland'

    elif str(nh)!='nan':
        if re.compile('san jose').search(nh) is not None:
            new_nh = 'san jose'
        else:
            new_nh = nh

    else:
        new_nh = nh

    return new_nh


## other cleaning? e.g. remove ", ca" from ends of neighborhoods


def chop_endings_function(thestring, list_of_endings):
    """
    Args:
        thestring: string to chop
        list_of_endings: list of endings to chop off of thestring
    Returns:
        thestring with ending chopped off, if it was there to begin with.
    """
    if type(thestring)==str:
        for ending in list_of_endings[::-1]:
            if thestring.endswith(ending):
                thestring = thestring[:-len(ending)].strip()

    return thestring



def lump_and_chop_neighborhoods(data):
    """
    Args:
        data: dataframe with columns ['neighborhood']
    Returns:
        dataframe with san francisco, oakland, and san jose neighborhoods lumped under city name, and ', ca' removed.

    functional dependencies: lump_neighborhoods_function, scrape_neighborhoods_from_wikipedia
    """

    data_copy = data.copy()

    sfhoods = scrape_neighborhoods_from_wikipedia('https://en.wikipedia.org/wiki/List_of_neighborhoods_in_San_Francisco')
    oakhoods = scrape_neighborhoods_from_wikipedia('https://en.wikipedia.org/wiki/List_of_neighborhoods_in_Oakland,_California')

    data_copy['neighborhood'] = data_copy['neighborhood'].apply(lump_neighborhoods_function, sfhoods = sfhoods, oakhoods = oakhoods)
    data_copy['neighborhood'] = data_copy['neighborhood'].apply(lambda nh: chop_endings_function(nh, ['ca', ', ca', ',ca']))

    return data_copy

def all_cleaning(raw_data):

    """
    Args:
        raw_data: includes columns ['description', 'price', 'neighborhood']
    Returns:
        dataframe with cleaned neighborhood information, no duplicates, and no rows with missing data.
    """

    out_df = raw_data.pipe(convert_street_addresses).pipe(fill_neighborhood_from_description).pipe(lump_and_chop_neighborhoods)
    out_df = out_df.drop_duplicates(subset = ['description', 'price'])
    out_df = out_df[['url', 'description', 'neighborhood', 'price', 'bedrooms']]
    out_df = out_df.dropna()

    #discard listings with truly enormous rent (probably actually for sale)
    out_df = out_df.loc[out_df['price']<1e4]

    #not using area as a feature now
    #dfclean['1000sqfeet']=dfclean['sq. feet']/1000  #normalize so all features are order 1 (no area is in 1000s of square feet)
    #no need to rescale price when only using one feature per neighborhood
    #dfclean['pricek']=dfclean['price']/1000 #normalize so all features are order 1 (price is in thousands of dollars)


    return out_df

def filter_to_popular_neighborhoods(df, min_count = 50):

    """
    narrow down neighborhoods considered. there are too many, with not enough samples for most.
    Args:
        df: cleaned dataframe with columns ['description', 'neighborhood']
        min_count: minimum number of listings a neighborhood must have to be counted. defaults to 50.
    Returns:
        rows of the input dataframe df where neighborhood is one with at least min_count listings.
    """


    byhood=df.groupby(by='neighborhood').count()['description'].sort_values(ascending=False)
    maxhoods=list(byhood[byhood >= 50].index) #pick off only neighborhoods with >50 listings

    df_maxhoods = df.copy()
    df_maxhoods['maxhood'] = df_maxhoods['neighborhood'].apply(lambda x: 1 if x in maxhoods else 0)
    df_maxhoods = df_maxhoods.loc[df_maxhoods['maxhood'] == 1]

    return df_maxhoods.drop(columns=['maxhood'])




# feature transformations

def transform_features(cleaned_data):

    """
    Args:
        cleaned_data: dataframe with columns ['price', 'bedrooms']
    Returns:
        input dataframe with an additional column, price per bedroom
    """

    transformed_data = cleaned_data.copy()
    transformed_data['price_per_room'] = transformed_data['price']/transformed_data['bedrooms']

    return transformed_data

# visualize

def plot_price_distr(transformed_data, neighborhoods = None):
    """
    Produce a histogram of the prices for the given neighborhood, along with the log normal of the same mean and standard deviation.
    Args:
        transformed_data: dataframe with columns ['neighborhood', 'price_per_room']
        neighborhoods: a list containing any of the values in the 'neighborhood' column of transformed_data
    Returns: Nothing. Just displays plots.
    """

    if neighborhoods == None:
         neighborhoods = list(set(transformed_data['neighborhood']))

    fig, axes = plt.subplots(len(neighborhoods), figsize=(10,50), sharex=True)
    plt.subplots_adjust(hspace = 1)


    for i, neighborhood in enumerate(neighborhoods):

        if type(axes) == np.ndarray:
            axi = axes[i]
        else:
            axi = axes

        # check neighborhood is represented in transformed_data:
        if neighborhood not in list(set(transformed_data['neighborhood'])):
            print(f"{neighborhood} not found as a value in the 'neighborhood' column of the given dataframe.")
            return None

        else:

            nh_df = transformed_data.loc[transformed_data['neighborhood'] == neighborhood]

            nh_prices = nh_df['price_per_room'].values

            nh_prices_ln = np.log([p for p in nh_prices if p > 0])  # exclude prices of 0
            mu_ln = np.mean(nh_prices_ln)
            std_ln = np.std(nh_prices_ln)


            axi.hist(nh_prices, bins = 10, density = True);
            Y = np.exp(np.arange(0, np.log(transformed_data['price_per_room'].max()), .01))
            Ydistr = (1/((2*np.pi)**.5*std_ln*Y))*np.array( np.exp(-(np.log(Y) - mu_ln)**2/(2*std_ln**2)) )
            axi.plot( Y, Ydistr  );

            axi.set_title(neighborhood);

#     return fig

# detect outliers

def get_outliers(transformed_data, neighborhoods = None, z = 2.5):
    """
    Detect listings in the dataframe transformed_data with anomologously low price per room.
    Args:
        transformed_data: dataframe with columns ['neighborhood', 'price_per_room']
        neighborhoods: a list containing any of the values in the 'neighborhood' column of transformed_data
        z: largest allowed z-score of a non-outlier. Defaults to 2.5.

    Returns: rows of transformed_data for the given neighborhoods with anomalously low prices per room.
    """

    if neighborhoods == None:
        neighborhoods = list(set(transformed_data['neighborhood']))

#     fig, axes = plt.subplots(len(neighborhoods))
#     plt.subplots_adjust(hspace = 1)

    low_outliers_df_all_neighborhoods = pd.DataFrame(columns = transformed_data.columns)

    for i, neighborhood in enumerate(neighborhoods):

#         if type(axes) == np.ndarray:
#             axi = axes[i]
#         else:
#             axi = axes

        # check neighborhood is represented in transformed_data:
        if neighborhood not in list(set(transformed_data['neighborhood'])):
            print(f"{neighborhood} not found as a value in the 'neighborhood' column of the given dataframe.")
            low_outliers_df = None

        else:

            nh_df = transformed_data.loc[transformed_data['neighborhood'] == neighborhood]

            nh_prices = nh_df['price_per_room'].values

            nh_prices_ln = np.log([p for p in nh_prices if p > 0])  # exclude prices of 0
            mu_ln = np.mean(nh_prices_ln)
            std_ln = np.std(nh_prices_ln)

            # Filter dataframe
            lowest_non_outlier_price = np.exp(mu_ln - z*std_ln)
            low_outliers_df = nh_df.loc[nh_df['price_per_room'] < lowest_non_outlier_price]


#             # Plotting
#             axi.hist(nh_prices, bins = 10, density = True);
#             Y = np.exp(np.arange(0,10,.01))
#             Ydistr = (1/((2*np.pi)**.5*std_ln*Y))*np.array( np.exp(-(np.log(Y) - mu_ln)**2/(2*std_ln**2)) )
#             axi.plot( Y, Ydistr  );
#             axi.set_title(neighborhood);

        low_outliers_df_all_neighborhoods = pd.concat([low_outliers_df_all_neighborhoods, low_outliers_df], axis = 0)


    return low_outliers_df_all_neighborhoods



#
