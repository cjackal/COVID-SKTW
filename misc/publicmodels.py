import requests
from io import BytesIO
from dateutil.parser import parse as dateparse
import re
import json
import zipfile
from bs4 import BeautifulSoup
import pandas as pd

with open('misc/po_code_state_map.json') as f:
    _po_code_state_map = json.load(f)
_st_to_fips = {v['state']:v['fips'] for v in _po_code_state_map}


def read_IHME(path=None, startdate=None):
    if path is None:
        r = BeautifulSoup(requests.get('http://www.healthdata.org/covid/data-downloads').text, "html5lib")
        for strong in r.find_all('strong'):
            if 'last updated' in strong.text:
                most_recent_date = ', '.join(strong.text.split(', ')[-2:-1])
                most_recent_date = str(dateparse(most_recent_date).date())
                url = 'https://ihmecovid19storage.blob.core.windows.net/latest/ihme-covid19.zip'
            elif startdate is not None:
                startdate = '-'.join((startdate[:4], startdate[4:6], startdate[6:])) if len(startdate)==8 else startdate
                for atag in strong.find_next_siblings('a'):
                    if str(dateparse(atag.text).date())<startdate:
                        most_recent_date = str(dateparse(atag.text).date())
                        url = atag['href']
                        if url[0]=='/': url = 'http://www.healthdata.org' + url

        inmemory = BytesIO(requests.get(url).content)

        with zipfile.ZipFile(inmemory) as ihme:
            for fpath in ihme.namelist():
                if 'Reference_hospitalization_all_locs.csv' in fpath:
                    with ihme.open(fpath) as csv:
                        df = pd.read_csv(csv)
                    break
    else:
        df = pd.read_csv(path)

    _states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
        'Colorado', 'Connecticut', 'Delaware', 'District of Columbia',
        'Florida', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
        'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
        'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
        'New Jersey', 'New Mexico', 'New York', 'North Carolina',
        'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
        'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
        'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
        'West Virginia', 'Wisconsin', 'Wyoming']

    df_usa = df[df['location_name'].isin(_states)][['date', 'location_name', 'deaths_mean_smoothed']]
    df_usa['fips'] = df_usa['location_name'].apply(lambda x: _st_to_fips[x])

    if path is None:
        df_usa.to_csv(f'data/ihme_{most_recent_date}.csv')
    else:
        df_usa.to_csv('data/ihme.csv')

def read_LANL(path=None, startdate=None):
    if path is None:
        lanl_meta = json.loads(requests.get('https://covid-19.bsvgateway.org/forecast/forecast_metadata.json').content)['us']
        if startdate is None:
            most_recent_date = lanl_meta['most_recent_date']
        else:
            startdate = '-'.join((startdate[:4], startdate[4:6], startdate[6:])) if len(startdate)==8 else startdate
            most_recent_date = ''
            for dt in sorted(lanl_meta['files'].keys()):
                if dt < startdate:
                    most_recent_date = dt
                else:
                    break
        relpath = lanl_meta['files'][most_recent_date]['quantiles_deaths']
        df = pd.read_csv('https://covid-19.bsvgateway.org'+relpath[1:])
    else:
        df = pd.read_csv(path)

    df_usa = df[['dates', 'q.50', 'state']].copy()
    df_usa['fips'] = df_usa['state'].apply(lambda x: _st_to_fips[x])

    if path is None:
        df_usa.to_csv(f'data/lanl_{most_recent_date}.csv')
    else:
        df_usa.to_csv('data/lanl.csv')

def read_MIT(path=None, startdate=None):
    if path is None:
        if startdate is None:
            url = f"https://api.github.com/repos/COVIDAnalytics/website/git/trees/master?recursive=1"
            r = requests.get(url)
            res = r.json()

            most_recent_date = '20200704'
            for file in res["tree"]:    
                if re.fullmatch(r'data/predicted/Global_V2_[0-9]{8}.csv', file["path"]):
                    if file["path"][-12:-4]>most_recent_date:
                        most_recent_date = file["path"][-12:-4]
        else:
            most_recent_date = startdate.replace('-', '')
        df = pd.read_csv(f'https://raw.githubusercontent.com/COVIDAnalytics/website/master/data/predicted/Global_V2_{most_recent_date}.csv')
    else:
        df = pd.read_csv(path)
    df_usa = df[df['Country']=='US'][['Province', 'Day', 'Total Detected Deaths']]
    df_usa = df_usa[df_usa['Province']!='None']
    df_usa['fips'] = df_usa['Province'].apply(lambda x: _st_to_fips[x])

    if path is None:
        df_usa.to_csv(f'data/mit_{most_recent_date[:4]}-{most_recent_date[4:6]}-{most_recent_date[6:]}.csv')
    else:
        df_usa.to_csv('data/mit.csv')

def read_YYG(path=None, startdate=None):
    pass

if __name__=='__main__':
    read_IHME(startdate='20200820')
    read_LANL(startdate='20200820')
    read_MIT(startdate='20200820')