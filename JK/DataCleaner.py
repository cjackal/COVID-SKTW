import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
if __name__ == '__main__':
    from utility import *
else:
    from .utility import *

homedir = get_homedir()

PATH_DEMO = f"{homedir}/data/us/aggregate_berkeley.csv"
PATH_GDP = f"{homedir}/JK/GDP.csv"
PATH_GEO = f"{homedir}/data/us/demographics/county_land_areas.csv"
PATH_MT = f"{homedir}/data/us/covid/nyt_us_counties_daily.csv"
PATH_MB = f"{homedir}/data/us/mobility/DL-us-mobility-daterow.csv"
PATH_SS = f"{homedir}/exploratory_HJo/seasonality_stateLevel.csv"
PATH_POL = f"{homedir}/JK/policy.csv"
PREPROCESSING = True

##################################################################################

FIPS_mapping, FIPS_full = get_FIPS(reduced=True)
oneweek = pd.Timedelta(days=7)
md_now = pd.Timestamp.now().strftime('%m%d')

"""
Read State-FIPS dictionary to be used in seasonality data.
"""
with open(f'{homedir}/JK/po_code_state_map.json') as f:
    po_st = json.load(f)

st_to_fips = {}
for dic in po_st:
    st_to_fips[dic['state']] = dic['fips']

"""
Read the datas.
"""
berkeley = pd.read_csv(PATH_DEMO, index_col=0)
berkeley['countyFIPS'] = berkeley['countyFIPS'].apply(correct_FIPS)
berkeley = fix_FIPS(berkeley, fipslabel='countyFIPS', reduced=True)

popularity_type= ['PopMale<52010',
    'PopFmle<52010', 'PopMale5-92010', 'PopFmle5-92010', 'PopMale10-142010',
    'PopFmle10-142010', 'PopMale15-192010', 'PopFmle15-192010',
    'PopMale20-242010', 'PopFmle20-242010', 'PopMale25-292010',
    'PopFmle25-292010', 'PopMale30-342010', 'PopFmle30-342010',
    'PopMale35-442010', 'PopFmle35-442010', 'PopMale45-542010',
    'PopFmle45-542010', 'PopMale55-592010', 'PopFmle55-592010',
    'PopMale60-642010', 'PopFmle60-642010', 'PopMale65-742010',
    'PopFmle65-742010', 'PopMale75-842010', 'PopFmle75-842010',
    'PopMale>842010', 'PopFmle>842010']
popularity_type_Male = popularity_type[::2]
popularity_type_Fmle = popularity_type[1::2]
motality_type = ['3-YrMortalityAge<1Year2015-17',
    '3-YrMortalityAge1-4Years2015-17', '3-YrMortalityAge5-14Years2015-17',
    '3-YrMortalityAge15-24Years2015-17',
    '3-YrMortalityAge25-34Years2015-17',
    '3-YrMortalityAge35-44Years2015-17',
    '3-YrMortalityAge45-54Years2015-17',
    '3-YrMortalityAge55-64Years2015-17',
    '3-YrMortalityAge65-74Years2015-17',
    '3-YrMortalityAge75-84Years2015-17', '3-YrMortalityAge85+Years2015-17']

demo = pd.DataFrame()
demo['fips'] = berkeley['countyFIPS']
demo['PopulationEstimate2018'] = berkeley['PopulationEstimate2018']
demo['PopRatioMale2017'] = berkeley['PopTotalMale2017'] / (berkeley['PopTotalMale2017']+berkeley['PopTotalFemale2017'])
demo['PopRatio65+2017'] = berkeley['PopulationEstimate65+2017'] / (berkeley['PopTotalMale2017']+berkeley['PopTotalFemale2017'])
demo['MedianAge,Male2010'] = berkeley['MedianAge,Male2010']
demo['MedianAge,Female2010'] = berkeley['MedianAge,Female2010']
demo['PopulationDensityperSqMile2010'] = berkeley['PopulationDensityperSqMile2010']
demo['MedicareEnrollment,AgedTot2017'] = berkeley['MedicareEnrollment,AgedTot2017'] / (berkeley['PopTotalMale2017']+berkeley['PopTotalFemale2017'])
demo['#Hospitals'] = 20000 * berkeley['#Hospitals'] / (berkeley['PopTotalMale2017']+berkeley['PopTotalFemale2017'])
demo['#ICU_beds'] = 10000 * berkeley['#ICU_beds'] / (berkeley['PopTotalMale2017']+berkeley['PopTotalFemale2017'])
for i in range(len(popularity_type_Male)):
    demo['PopRatio'+popularity_type_Male[i][3:]] = berkeley[popularity_type_Male[i]] / (berkeley[popularity_type_Male[i]]+berkeley[popularity_type_Fmle[i]])
    demo['PopRatio'+popularity_type_Male[i][7:]] = berkeley[popularity_type_Male[i]] / (berkeley['PopTotalMale2017']+berkeley['PopTotalFemale2017'])
demo['HeartDiseaseMortality'] = berkeley['HeartDiseaseMortality']
demo['StrokeMortality'] = berkeley['StrokeMortality']
demo['DiabetesPercentage'] = berkeley['DiabetesPercentage']
demo['Smokers_Percentage'] = berkeley['Smokers_Percentage']
demo['#EligibleforMedicare2018'] = berkeley['#EligibleforMedicare2018']
demo['mortality2015-17Estimated'] = berkeley['mortality2015-17Estimated']

demo.fillna(0, inplace=True)

gdp = pd.read_csv(PATH_GDP)
gdp['fips'] = gdp['fips'].apply(correct_FIPS)
gdp = fix_FIPS(gdp, fipslabel='fips', reduced=True)

geo = pd.read_csv(PATH_GEO, usecols=[0,2,3,4,5])
geo['County FIPS'] = geo['County FIPS'].apply(correct_FIPS)
geo = fix_FIPS(geo, fipslabel='County FIPS', reduced=True)

motality = pd.read_csv(PATH_MT, parse_dates=['date'])
motality.dropna(inplace=True)
motality['fips'] = motality['fips'].apply(correct_FIPS)
motality = fix_FIPS(motality, fipslabel='fips', datelabel='date', reduced=True)

mobility = pd.read_csv(PATH_MB, parse_dates=['date'])
mobility.dropna(subset=['fips'], inplace=True)
mobility['fips'] = mobility['fips'].apply(correct_FIPS)
mobility.drop(columns=['admin_level', 'samples'], inplace=True)
mobility = fix_FIPS(mobility, fipslabel='fips', datelabel='date', reduced=True)

seasonality = pd.read_csv(PATH_SS, index_col=0, parse_dates=['date'])
seasonality['date'] += pd.Timedelta(days = 365*3)
seasonality.replace({'state':st_to_fips}, inplace=True)
seasonality.replace({'state':{'New York City':'36061'}}, inplace=True)

policy = pd.read_csv(PATH_POL, parse_dates=['date'])
policy['state'] = policy['state'].apply(lambda x:'0'*(2-len(str(x)))+str(x))
policy['fips'] = policy['fips'].apply(correct_FIPS)
policy.replace({'fips':FIPS_mapping}, inplace=True)

FIPS_demo = set(demo['fips']); FIPS_gdp = set(gdp['fips']); FIPS_mt = set(motality['fips']); FIPS_mb = set(mobility['fips'])

date_st_mt = motality['date'].min(); date_ed_mt = motality['date'].max()
date_st_mb = mobility['date'].min(); date_ed_mb = mobility['date'].max()

"""
Filling in missing dates by searching closest date of the same day.
"""
ndays = (date_ed_mb - date_st_mb).days+1
dwin = pd.date_range(start=date_st_mb, end=date_ed_mb)
altrange = [item for sublist in [[n,-n] for n in range(1, ndays//7+1)] for item in sublist]

m50toAdd = []
for fips in FIPS_mb:
    df = mobility[mobility['fips']==fips]
    if len(df) != ndays:
        existingdates = list(df['date'])
        missingdates = set(dwin).difference(set(existingdates))
        for dt in missingdates:
            samedays = [dt + n*oneweek for n in altrange if (dt + n*oneweek) in existingdates]
            if samedays:
                m50, m50_index = df[df['date']==samedays[0]]['m50'].iloc[0], df[df['date']==samedays[0]]['m50_index'].iloc[0]
            else:
                m50, m50_index = df[df['date']==existingdates[-1]]['m50'].iloc[0], df[df['date']==existingdates[-1]]['m50_index'].iloc[0]
            m50toAdd.append([dt, fips, m50, m50_index])
mobility = mobility.append(pd.DataFrame(m50toAdd, columns=mobility.columns))

"""
Filling in missing counties using their state.
"""
for fips in FIPS_demo.difference(FIPS_mb):
    stt = str(int(fips[:2]))
    if stt in FIPS_mb:
        dummy = mobility[mobility['fips']==stt].copy()
        dummy.loc[:,'fips'] = fips
        mobility = mobility.append(dummy)
FIPS_mb = set(mobility['fips'])

"""
Save preprocessed dataframes.
"""
try:
    os.mkdir(f'{homedir}/JK/preprocessing/{md_now}')
except OSError as error:
    print(error)

demo.to_csv(f'{homedir}/JK/preprocessing/{md_now}/demographic.csv', index=False)
motality.to_csv(f'{homedir}/JK/preprocessing/{md_now}/motality.csv', index=False)
mobility.to_csv(f'{homedir}/JK/preprocessing/{md_now}/mobility.csv', index=False)

# settings
date_st = max(date_st_mt, date_st_mb)
date_ed = min(date_ed_mt, date_ed_mb)
date_win = pd.date_range(start=date_st, end=date_ed)

columns_demo = list(demo.columns); columns_demo.remove('fips')
columns_gdp = list(gdp.columns); columns_gdp.remove('fips')
columns_geo = list(geo.columns); columns_geo.remove('County FIPS')
columns_mt = ['cases', 'deaths']
columns_mb = ['m50', 'm50_index']
columns_ss = ['seasonality']
columns_pol = ['emergency', 'safeathome', 'business']

with open(f'{homedir}/JK/preprocessing/{md_now}/columns_ctg.txt', 'w') as f:
    print(columns_demo+columns_gdp+columns_geo, file=f)
with open(f'{homedir}/JK/preprocessing/{md_now}/columns_ts.txt', 'w') as f:
    print(columns_mt+columns_mb+columns_ss+columns_pol, file=f)

print('# Demographic FIPS=', len(FIPS_demo), ', # Motality FIPS=', len(FIPS_mt), ', # Mobility FIPS=', len(FIPS_mb))
print('First date to be trained:', date_st, ', Final date to be trained:', date_ed)

"""
Generate training data
"""
if PREPROCESSING:
    data_ts = []
    data_ctg = []
    counter = 0
    for fips in sorted(FIPS_demo):
        counter += 1
        if counter % 300 == 0:
            print('.', end='')
        data1 = demo[demo['fips']==fips][columns_demo].to_numpy()[0]
        
        data2 = motality[(motality['fips']==fips) & (motality['date'].isin(date_win))][['date']+columns_mt]
        _ = [[dt, 0, 0] for dt in date_win if dt not in list(data2['date'])]
        data2 = data2.append(pd.DataFrame(_, columns=['date']+columns_mt))
        data2 = data2.sort_values(by=['date'])[columns_mt].to_numpy()
        
        data3 = mobility[(mobility['fips']==fips) & (mobility['date'].isin(date_win))][['date']+columns_mb]
        data3 = data3.sort_values(by=['date'])[columns_mb].to_numpy()

        if fips == '36061':             # New York City
            data4 = seasonality[(seasonality['state']==fips) & (seasonality['date'].isin(date_win))][['date']+columns_ss]
        else:
            data4 = seasonality[(seasonality['state']==fips[:2]) & (seasonality['date'].isin(date_win))][['date']+columns_ss]
        data4 = data4.sort_values(by=['date'])[columns_ss].to_numpy()

        data5 = gdp[gdp['fips']==fips][columns_gdp].to_numpy()[0]

        data6 = []
        _ = policy[policy['state']==fips[:2]].copy()
        _ = _[(_['fips']=='0')|(_['fips']==fips)][['date']+columns_pol]
        _.drop_duplicates(subset='date', keep='last', inplace=True)
        _.reset_index(drop=True, inplace=True)
        for dt in date_win:
            if len(_[_['date']<=dt])==0:
                data6.append([0,0,0])
            else:
                data6.append(list(_[_['date']<=dt].iloc[-1][columns_pol].apply(int)))
        data6 = np.asarray(data6)

        data7 = geo[geo['County FIPS']==fips][columns_geo].to_numpy()[0]

        data_ctg.append(np.hstack((data1, data5, data7)))
        data_ts.append(np.hstack((data2, data3, data4, data6)))
    np.save(f'{homedir}/JK/preprocessing/{md_now}/data_ctg.npy', np.asarray(data_ctg, dtype=np.float32))
    np.save(f'{homedir}/JK/preprocessing/{md_now}/data_ts.npy', np.asarray(data_ts, dtype=np.float32))
    with open(f'{homedir}/JK/preprocessing/{md_now}/FIPS.txt', 'w') as f:
        print(sorted(FIPS_demo), file=f)
    with open(f'{homedir}/JK/preprocessing/{md_now}/date_ed.txt', 'w') as f:
        print(date_ed.strftime('%Y-%m-%d'), file=f)
    print('Preprocessing finished.')