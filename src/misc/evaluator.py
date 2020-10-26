import os
import sys
import json
import numpy as np
import pandas as pd
from .utility import get_homedir, correct_FIPS, gen_frame

homedir = get_homedir()
datadir = os.path.join(homedir, 'data')
PATH_MT = os.path.join(datadir, "nyt_us_counties_daily.csv")

def _get_date(x):
    return '-'.join(x.split('-')[:3])

def _get_fips(x):
    return x.split('-')[-1]

def pinball_loss(y_true, y_pred, quantile=0.5):
    import numpy as np
    delta = y_true - y_pred
    loss_above = np.sum(delta[delta>0])*(quantile)
    loss_below = np.sum(-1*delta[delta<0])*(1-quantile)
    return (loss_above + loss_below) / len(y_true)

def evaluate(test_df, user_df, target='deaths'):
    join_df = test_df.join(user_df, how = 'inner')
    if (len(join_df) != len(test_df)):
        raise Exception("Submission not right length")
    if (user_df.isna().sum().sum() > 0 ):
        raise Exception("Submission Contains NaN.")
    if (join_df.index.equals(test_df.index) == False):
        raise Exception("Incorrect ID format.")
    total_loss = 0
    for column in ['10','20','30','40','50', '60', '70', '80', '90']:
        quantile = int(column) / 100.0
        total_loss += pinball_loss(join_df[target].values, join_df[column].values, quantile) / 9.0
    return total_loss

def get_score(df_to_score, start_date, end_date=None, all_fips=None, target='deaths'):
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.strftime('%Y-%m-%d')

    daily_df = pd.read_csv(PATH_MT)
    daily_df['fips'] = daily_df['fips'].astype(int)
    if end_date is None:
        end_date = daily_df['date'].max()
    daily_df['id'] = daily_df['date'] + '-' + daily_df['fips'].astype(str)
    preperiod_df = daily_df[daily_df['date']<start_date]
    daily_df = daily_df[(daily_df['date']<=end_date) & (daily_df['date']>=start_date)]

    sample_submission = gen_frame(start_date, end_date)
    sample_submission['date'] = sample_submission['id'].apply(_get_date)
    sample_submission['fips'] = sample_submission['id'].apply(_get_fips).astype(int)

    # Disabled FIPS is a set of FIPS to avoid scoring. Covid_active_fips is where there has been reports of covid, 
    # and inactive_fips are fips codes present in sample submission but with no cases reported by the New York Times.
    # New_active_fips are FIPS that were introduced into the dataset during the scoring period. 
    # Active FIPS should be scored against deaths data from NYT if such data is available, 
    # but Inactive FIPS should be scored with a target of 0.
    disabled_fips = set({
        ## NEW YORK
        36005, 36047, 36081, 36085, 
        ## Peurto Rico
        72001, 72003, 72005, 72007, 72009, 72011, 72013, 72015, 72017,
        72019, 72021, 72023, 72025, 72027, 72029, 72031, 72033, 72035,
        72037, 72039, 72041, 72043, 72045, 72047, 72049, 72051, 72053,
        72054, 72055, 72057, 72059, 72061, 72063, 72065, 72067, 72069,
        72071, 72073, 72075, 72077, 72079, 72081, 72083, 72085, 72087,
        72089, 72091, 72093, 72095, 72097, 72099, 72101, 72103, 72105,
        72107, 72109, 72111, 72113, 72115, 72117, 72119, 72121, 72123,
        72125, 72127, 72129, 72131, 72133, 72135, 72137, 72139, 72141,
        72143, 72145, 72147, 72149, 72151, 72153,
        ## Virgin Islands
        78010, 78020, 78030
    })
    prev_active_fips = set(preperiod_df.fips.unique())
    curr_active_fips = set(daily_df.fips.unique())
    if all_fips is None:
        all_fips = set(sample_submission.fips.unique())
    elif type(all_fips) is list:
        all_fips = set(all_fips)
    covid_active_fips = prev_active_fips.intersection(all_fips).intersection(curr_active_fips) - disabled_fips
    inactive_fips = all_fips - prev_active_fips - curr_active_fips - disabled_fips
    new_active_fips = (curr_active_fips - prev_active_fips).intersection(all_fips) - disabled_fips

    # Create a DataFrame of all 0's for inactive fips by getting those from sample submission.
    inactive_df = sample_submission.set_index('fips')[['id','50']].loc[inactive_fips]
    inactive_df = inactive_df.set_index('id').rename({'50':target}, axis=1)
    assert inactive_df.sum().sum() == 0
    # Create a DataFrame of active fips from the New York Times data
    active_df = daily_df.set_index('fips')[['id', target]].loc[covid_active_fips].set_index('id')

    # Create dataframe for new fips
    sample_search = sample_submission.set_index('fips')[['id','50']].rename({'50':target}, axis=1)
    daily_search = daily_df.set_index('fips')
    new_df_arr = []
    for fips in new_active_fips:
        tmp_sample = sample_search.loc[[fips]].set_index('id')
        tmp_daily = daily_search.loc[[fips]].set_index('id')
        tmp_sample.update(tmp_daily)
        tmp_sample = tmp_sample[tmp_sample.index<=tmp_daily.index.max()]
        new_df_arr.append(tmp_sample)

    # Join the data frames
    example = None
    if (len(new_active_fips) > 0):
        new_df = pd.concat(new_df_arr)
        example = pd.concat([inactive_df, active_df, new_df]).sort_index()
    else:
        example = pd.concat([inactive_df, active_df]).sort_index()
    # Read some CSV for score
    df = df_to_score.set_index('id').sort_index()
    return evaluate(example[[target]], df, target=target)

def get_score_countywise(df_to_score, date_st=None, date_ed=None, target='deaths'):
    plot_df = df_to_score.copy()
    plot_df['date'] = plot_df['id'].apply(lambda x: '-'.join(x.split('-')[:3]))
    plot_df['fips'] = plot_df['id'].apply(lambda x: correct_FIPS(x.split('-')[-1]))
    if date_st is None:
        date_st = plot_df['date'].min()
    if date_ed is None:
        date_ed = plot_df['date'].max()
    plot_df = plot_df[(plot_df['date']>=date_st) & (plot_df['date']<=date_ed)]
    plot_df.reset_index(inplace=True, drop=True)

    daily_df = pd.read_csv(PATH_MT)
    daily_df['fips'] = daily_df['fips'].astype(int)
    daily_df['id'] = daily_df['date'] + '-' + daily_df['fips'].astype(str)
    daily_df['fips'] = daily_df['fips'].apply(correct_FIPS)
    daily_df = daily_df[(daily_df['date']<=date_ed) & (daily_df['date']>=date_st)]

    plot_df = plot_df.join(daily_df[['id', target]].set_index('id'), on='id', how='inner')
    plot_df['pinball_loss'] = 0
    for i in range(1, 10):
        plot_df['pinball_loss'] += (plot_df[target] - plot_df[str(10*i)]).apply(lambda x: max(i*x, (i-10)*x)/90)

    return plot_df.drop(columns=target)