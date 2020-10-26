import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation, performance_metrics
from .utility import get_homedir

homedir = get_homedir()
datadir = os.path.join(homedir, 'data')

# function for getting the first day of the week
# Prophet requires daily data, so we will just transform from week to day
def prophet_getFirstDayofWeek(yr, wk):
    d = "%s-W%s" % (yr, wk)
    # The -1 and -%w pattern tells the parser to pick the Monday in that week
    firstdayofweek = datetime.datetime.strptime(d + '-1', "%Y-W%W-%w")
    return firstdayofweek

# Prophet also requires dates to be stored in a column called 'ds'
# And that the values to predict will be stored in a column called 'y'
def prophet_changeDfStructure(df, y_name, virusType = None):
    if virusType != None:
        df = df.loc[df['Virus'].isin(virusType)]
    # Remove rows with week>=54, which errors out
    df = df[df['WEEK']<54].copy()
    df["ds"] = df.apply(lambda x: prophet_getFirstDayofWeek(x.YEAR, x.WEEK), axis=1)
    df["y"] = df[y_name]
    return df

if __name__=='__main__':
    # define df_PI, which includes a total number of cases of PI in US
    df_PI = pd.read_csv(os.path.join(datadir, "statewide_pi_deaths_2012_2020.csv"))
    df_PI["YEAR"] = df_PI.apply(lambda x: int(x.SEASON[0:4]), axis=1)
    df_PI.loc[df_PI['WEEK']<=26, 'YEAR'] += 1

    # change the structure of df_PI for prophet
    df_PI = prophet_changeDfStructure(df_PI, "PERCENT P&I")

    # get the list of states
    stateList = df_PI['SUB AREA'].unique()

    # train prophet for each state, and make predictions
    m_PI_eachState = []
    forecast_eachState = []
    for state in stateList:
        # select state
        df_temp = df_PI.loc[df_PI['SUB AREA']==state]
        
        # train prophet
        m_PI = Prophet(changepoint_prior_scale=0.01, seasonality_mode='multiplicative')
        m_PI.fit(df_temp)
        
        # make prediction
        future = m_PI.make_future_dataframe(periods=90)
        forecast = m_PI.predict(future)

        # save into list
        m_PI_eachState.append(m_PI)
        forecast_eachState.append(forecast)

    df = pd.DataFrame(columns=['state', 'date', 'seasonality'])
    for i, state in enumerate(m_PI_eachState):
        fig = m_PI_eachState[i].plot_components(forecast_eachState[i])
        ax = plt.gca()
        line = ax.lines[0]
        y_test = line.get_ydata()
        x_test = line.get_xdata()

        df_temp = pd.DataFrame(data={'state': state, 'date': x_test, 'seasonality': y_test})
        df = pd.concat([df, df_temp], ignore_index=True)
    df.to_csv(os.path.join(datadir, 'seasonality_stateLevel.csv'))