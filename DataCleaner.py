import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from misc.utility import *

homedir = get_homedir()
logger = logging.getLogger('main.DataCleaner')

def DataCleaner(config_name, tmp, ver='frozen'):
    with open(config_name, 'r') as f:
        config_dict = json.load(f)
    PATH_SCR = os.path.join(homedir, "LSTM/preprocessing", tmp)

    config_st = pd.Timestamp(config_dict['start_date'])
    if config_dict['end_date'] is None:
        config_ed = config_st + pd.Timedelta(days=13)
    else:
        config_ed = pd.Timestamp(config_dict['end_date'])
    if config_dict['start_train'] is None:
        start_train = None
    else:
        start_train = pd.Timestamp(config_dict['start_train'])
    end_train = pd.Timestamp(config_dict['end_train'])

    PATH_DEMO = os.path.join(homedir, "LSTM/aggregate_berkeley.csv")
    PATH_GDP = os.path.join(homedir, "LSTM/GDP.csv")
    PATH_GEO = os.path.join(homedir, "LSTM/county_land_areas.csv")
    PATH_MT = os.path.join(homedir, "data/nyt_us_counties_daily.csv")
    PATH_MB = os.path.join(homedir, "data/DL-us-mobility-daterow.csv")
    PATH_SS = os.path.join(homedir, "LSTM/seasonality_stateLevel.csv")
    PATH_POL = os.path.join(homedir, "data/state_policy.csv")
    ############################################################################

    FIPS_mapping, FIPS_full = get_FIPS(reduced=True)
    oneweek = pd.Timedelta(days=7)
    md_now = pd.Timestamp.now().strftime('%m%d')

    """
    Generate base DataFrame in the format of sample_submission.
    """
    logger.debug("Generate base DataFrame.")
    gen_submission(config_st, config_ed, file=False).to_csv(os.path.join(PATH_SCR, 'sample_submission.csv'), index=False)

    """
    Read State-FIPS dictionary to be used in seasonality data.
    """
    with open(os.path.join(homedir, 'misc/po_code_state_map.json')) as f:
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
    policy = policy[policy['policy_type'].isin(['ShelterInPlace', 'StateOfEmergency', 'Non-essentialBusinesses'])]
    policy['fips_code'] = policy['fips_code'].apply(correct_FIPS)
    policy.replace({'fips_code':FIPS_mapping}, inplace=True)

    FIPS_demo = set(demo['fips']); FIPS_gdp = set(gdp['fips']); FIPS_mt = set(motality['fips']); FIPS_mb = set(mobility['fips'])

    date_st_mt = motality['date'].min(); date_ed_mt = motality['date'].max()
    date_st_mb = mobility['date'].min(); date_ed_mb = mobility['date'].max()

    """
    Filling in missing dates by searching closest date of the same day.
    """
    ndays = (max(date_ed_mb, date_ed_mt) - date_st_mb).days+1
    dwin = pd.date_range(start=date_st_mb, end=max(date_ed_mb, date_ed_mt))
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

    # settings
    if start_train is None:
        date_st = max(date_st_mt, date_st_mb)
    else:
        date_st = max(date_st_mt, date_st_mb, start_train)
    date_ed = min(end_train, max(date_ed_mt, date_ed_mb))
    date_win = pd.date_range(start=date_st, end=date_ed)

    columns_ctg = []
    columns_ts = []
    columns_demo = list(demo.columns); columns_demo.remove('fips'); columns_ctg += columns_demo
    columns_gdp = list(gdp.columns); columns_gdp.remove('fips'); columns_ctg += columns_gdp
    columns_geo = list(geo.columns); columns_geo.remove('County FIPS'); columns_ctg += columns_geo
    columns_mt = ['cases', 'deaths']; columns_ts += columns_mt
    columns_mb = ['m50', 'm50_index']; columns_ts += columns_mb
    columns_ss = ['seasonality']; columns_ts += columns_ss
    columns_pol = ['emergency', 'safeathome', 'business']
    if ver=='frozen': columns_ts += columns_pol
    else: columns_ts += ['isweekend']

    with open(os.path.join(PATH_SCR, 'columns_ctg.txt'), 'w') as f:
        print(columns_ctg, file=f)
    with open(os.path.join(PATH_SCR, 'columns_ts.txt'), 'w') as f:
        print(columns_ts, file=f)

    logger.info('Categorical features: '+', '.join(columns_ctg))
    logger.info('Timeseries features: '+', '.join(columns_ts))
    logger.info(f'# Demographic FIPS={len(FIPS_demo)}, # Motality FIPS={len(FIPS_mt)}, # Mobility FIPS={len(FIPS_mb)}')
    logger.info(f'First date to be trained: {date_st}, Final date to be trained: {date_ed}')

    """
    Generate training data
    """
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
        _ = policy[(policy['fips_code']==fips)|(policy['fips_code']==fips[:2]+'000')]
        df = _[_['date']<date_win[0]]
        emergency, safeathome, business = 0, 0, 0
        for i in df.index:
            if df.loc[i, 'policy_type']=='StateOfEmergency':
                emergency = int(df.loc[i, 'start_stop'])
            elif df.loc[i, 'policy_type']=='ShelterInPlace':
                safeathome = int(df.loc[i, 'start_stop'])
            else:
                business = int(df.loc[i, 'start_stop'])
        for dt in date_win:
            df = _[_['date']==dt]
            for i in df.index:
                if df.loc[i, 'policy_type']=='StateOfEmergency':
                    emergency = int(df.loc[i, 'start_stop'])
                elif df.loc[i, 'policy_type']=='ShelterInPlace':
                    safeathome = int(df.loc[i, 'start_stop'])
                else:
                    business = int(df.loc[i, 'start_stop'])
            data6.append([emergency, safeathome, business])
        data6 = np.asarray(data6)

        data7 = geo[geo['County FIPS']==fips][columns_geo].to_numpy()[0]

        isweekend = np.asarray([(_.dayofweek)//5 for _ in date_win])[:, np.newaxis]

        data_ctg.append(np.hstack((data1, data5, data7)))
        if ver=='frozen':
            data_ts.append(np.hstack((data2, data3, data4, data6)))
        else:
            data_ts.append(np.hstack((data2, data3, data4, isweekend)))
    np.save(os.path.join(PATH_SCR, 'data_ctg.npy'), np.asarray(data_ctg, dtype=np.float32))
    np.save(os.path.join(PATH_SCR, 'data_ts.npy'), np.asarray(data_ts, dtype=np.float32))
    with open(os.path.join(PATH_SCR, 'FIPS.txt'), 'w') as f:
        print(sorted(FIPS_demo), file=f)
    with open(os.path.join(PATH_SCR, 'date_ed.txt'), 'w') as f:
        print(date_ed.strftime('%Y-%m-%d'), file=f)
    logger.info('Preprocessing complete.')
    with open(os.path.join(PATH_SCR, 'config.json'), 'w') as f:
        json.dump(config_dict, f)
