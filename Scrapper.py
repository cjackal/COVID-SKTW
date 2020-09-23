import os
import requests
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from misc.utility import get_homedir

homedir = get_homedir()
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
        # Fill in missing rows with no new cases/deaths, as it appears that these missing days
        # have no new cases/deaths.
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
    logging.info('Download DL mobility data.')
    req = requests.get("https://raw.githubusercontent.com/descarteslabs/DL-COVID-19/master/DL-us-mobility-daterow.csv")
    with open(os.path.join(homedir, 'data/DL-us-mobility-daterow.csv'), 'wb') as f:
        f.write(req.content)

    logging.info('Download and preprocess NYT motality data.')
    nyt_df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv', parse_dates=['date'])

    nyt_df.loc[nyt_df['county'] == 'New York City', 'fips'] = 36061
    nyt_df.loc[nyt_df['state'] == 'Guam', 'fips'] = 66010
    nyt_df = nyt_df[nyt_df['fips'].isna() == False]
    nyt_df['fips'] = nyt_df['fips'].astype(int)
    nyt_df.set_index('fips', inplace=True)
    fips_total = nyt_df.index.unique()
    logging.debug(f'# of counties: {len(fips_total)}')

    df_list = []
    logging.debug('Filling out missing dates.')
    for i, fips in tqdm(enumerate(fips_total), desc='Processing NYT motality'):
        df = nyt_df.loc[[fips]]
        df = process_fips_df(df)
        df = difference_fips_df(df)
        df_list.append(df)
        if (i+1) % 200 == 0:
            logging.debug('Process done: {i * 200}/{len(fips_total)}')
    pd.concat(df_list).to_csv(os.path.join(homedir, 'data/nyt_us_counties_daily.csv'))

    now = pd.Timestamp.utcnow().strftime("%Y%m%d")
    with open(os.path.join(homedir, 'data/date.txt'), 'w') as f:
        print(now, file=f)
