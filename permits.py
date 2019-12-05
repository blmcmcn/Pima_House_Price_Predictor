import pandas as pd
import numpy as np
from selenium import webdriver
from tqdm import tqdm
import pdb

def scrape():

    # url for the permit search site
    url = "https://permits.pima.gov/acaprod/APO/APOLookup.aspx?TabName=APO"

    # load the data
    data = pd.read_csv("data/permits.csv")
    data.fillna(value='filler', inplace=True)

    for n in range(data.shape[0]):
        if not data.at[n,'parcel'] == 'filler': continue
        print(f"\n{n}/{data.shape[0]}")
        # open up the browsing session in Chrome
        driver = webdriver.Chrome()
        # set the implicit wait time (seconds)
        driver.implicitly_wait(10)
        # navigate to url
        driver.get(url)
        # select search box
        search = driver.find_element_by_class_name('gs_search_box')
        # make sure there are not previous results in the search box
        search.send_keys('')
        # enter in the record number
        search.send_keys(data.iloc[n]['Record Number'])
        # click the search
        driver.find_element_by_class_name('gs_go_img').click()
        # click on the 'more details' icon
        try:
            driver.find_element_by_id("imgMoreDetail").click()
            # click on the 'parcel information' icon
            driver.find_element_by_id("imgParcel").click()
        except:
            driver.close()
            continue
        # get the parcel number
        parcelN = driver.find_element_by_xpath('//*[@id="ctl00_PlaceHolderMain_PermitDetailList1_palParceList"]/div/div').text
        # add the parcel to the data
        data.at[n,'parcel'] = parcelN
        # close browser
        driver.close()
        # save data
        print(f"saved {parcelN} to {n}")
        data.to_csv("data/permits.csv", index=False)
