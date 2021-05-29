import os
import requests
import logging
import json
import numpy as np
import pandas as pd
from io import BytesIO
from bs4 import BeautifulSoup
from .misc.utility import pbar, get_homedir, correct_FIPS

homedir = get_homedir()
datadir = os.path.join(homedir, 'data')
sourcedir = os.path.join(homedir, 'src')
logger = logging.getLogger('main.Scrapper')

def add_missing_date_rows(df, row, low_date, max_date):
    day_delta = np.timedelta64(1,'D')
    day = low_date + day_delta
    while day < max_date:
        copy_row = row.copy()
        copy_row['date'] = day
        df = df.append(copy_row)
        day = day + day_delta
    return df

def process_fips_df(df):
    # Ensures the data is sorted by date.
    df = df.sort_values('date')

    dates = df['date'].values
    for i in range(1, len(dates)):
        # Fill in missing rows with no new cases/deaths, as it appears that
        # these missing days have no new cases/deaths.
        if(dates[i] > dates[i-1]+np.timedelta64(1,'D')):
            return process_fips_df(add_missing_date_rows(df, df.iloc[i], dates[i-1], dates[i]))
    return df

def difference_fips_df(df):
    # Ensures the data is sorted by date.
    df = df.sort_values('date')

    initial_deaths = df['deaths'].values[0]
    initial_cases = df['cases'].values[0]
    new_cases = df['cases'].diff().values
    new_cases[0] = initial_cases
    # Must be clipped as on occasion negative values appear due to bad data.
    df['cases'] = np.clip(new_cases, 0, np.inf)
    new_deaths = df['deaths'].diff().values
    new_deaths[0] = initial_deaths
    # Must be clipped as on occasion negative values appear due to bad data.
    df['deaths'] = np.clip(new_deaths, 0, np.inf)
    return df

def Scrapper():
    logger.info('Download DL mobility data.')
    req = requests.get("https://raw.githubusercontent.com/descarteslabs/DL-COVID-19/master/DL-us-mobility-daterow.csv")
    with open(os.path.join(datadir, 'DL-us-mobility-daterow.csv'), 'wb') as f:
        f.write(req.content)

    logger.info('Download and preprocess NYT motality data.')
    nyt_df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv', parse_dates=['date'])

    nyt_df.loc[nyt_df['county'] == 'New York City', 'fips'] = 36061
    nyt_df.loc[nyt_df['state'] == 'Guam', 'fips'] = 66010
    nyt_df = nyt_df[nyt_df['fips'].isna() == False]
    nyt_df['fips'] = nyt_df['fips'].astype(int)
    nyt_df.set_index('fips', inplace=True)
    fips_total = nyt_df.index.unique()
    logger.debug(f'# of counties: {len(fips_total)}')

    dfs = []
    logger.debug('Filling out missing dates.')
    for i, fips in pbar(enumerate(fips_total), desc='Processing NYT motality',
        total=len(fips_total),bar_format='{l_bar}{bar}[{elapsed}<{remaining}]'):
        df = nyt_df.loc[[fips]]
        df = process_fips_df(df)
        df = difference_fips_df(df)
        dfs.append(df)
    pd.concat(dfs).to_csv(os.path.join(datadir, 'nyt_us_counties_daily.csv'))

    logger.info('Download and preprocess HHS policy data.')
    # r = BeautifulSoup(requests.get('https://healthdata.gov/dataset/covid-19-state-and-county-policy-orders').text, "html5lib")
    # url = r.find_all("a", class_="data-link")[0]['href']
    req = requests.get('https://healthdata.gov/api/views/gyqz-9u7n/rows.csv?accessType=DOWNLOAD')

    with open(os.path.join(sourcedir, 'misc', 'po_code_state_map.json')) as f:
        po_code_state_map = json.load(f)
    pc_to_fips = {v['postalCode']: int(''.join((v['fips'], '000')))
                     for v in po_code_state_map}

    hhs_df = pd.read_csv(BytesIO(req.content))
    hhs_df['date'] = hhs_df['date'].apply(lambda x: '20'+x[2:] if x[:2]=='00' else x)
    hhs_df['date'] = hhs_df['date'].apply(pd.to_datetime)
    hhs_df['policy_type'] = hhs_df['policy_type'].apply(lambda x: ''.join([s.capitalize() for s in x.split(' ')]))
    hhs_df['start_stop'] = hhs_df['start_stop'].apply(lambda x: (x=='start'))
    for i in hhs_df.index:
        if hhs_df.loc[i, 'policy_level']=='state':
            hhs_df.loc[i, 'fips_code'] = pc_to_fips.get(hhs_df.loc[i, 'state_id'], hhs_df.loc[i, 'fips_code'])
    hhs_df.dropna(subset=['fips_code'], inplace=True)
    hhs_df['fips_code'] = hhs_df['fips_code'].apply(correct_FIPS)
    hhs_df = hhs_df[['fips_code', 'date', 'policy_type', 'start_stop']]
    hhs_df = hhs_df.sort_values('date')
    hhs_df.to_csv(os.path.join(datadir, 'state_policy.csv'), index=False)

    logger.info('Download and preprocess CDC vaccination data.')
    req = requests.get('https://data.cdc.gov/api/views/unsk-b7fc/rows.csv?accessType=DOWNLOAD')
    cdc_df = pd.read_csv(BytesIO(req.content), parse_dates=['Date'])[['Date', 'Location', 'Admin_Per_100k_12Plus', 'Admin_Per_100k_18Plus','Admin_Per_100k_65Plus']]
    cdc_df = cdc_df[cdc_df['Location'].isin(set(pc_to_fips))]
    cdc_df['fips_code'] = cdc_df['Location'].apply(lambda x: pc_to_fips.get(x,'XX'))
    cdc_df = cdc_df.sort_values('Date')
    cdc_df.to_csv(os.path.join(datadir, 'vaccination.csv'), index=False)

    now = pd.Timestamp.utcnow().strftime("%Y%m%d")
    with open(os.path.join(datadir, 'date.txt'), 'w') as f:
        print(now, file=f)