import os
import json
import numpy as np
import plotly.express as px
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from .misc.utility import get_homedir, correct_FIPS, fix_FIPS
from .LSTM import SingleLayerConditionalRNN, load_Dataset, train_full, get_StandardScaler, normalizer

homedir = get_homedir()
PATH_DEMO = os.path.join(homedir, "data/aggregate_berkeley.csv")
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
demo.set_index('fips', inplace=True)
demo.fillna(0, inplace=True)

PATH_GDP = os.path.join(homedir, "data/GDP.csv")
gdp = pd.read_csv(PATH_GDP)
gdp['fips'] = gdp['fips'].apply(correct_FIPS)
gdp = fix_FIPS(gdp, fipslabel='fips', reduced=True)
gdp.set_index('fips', inplace=True)
################################################################################

config = {
    'midMay': {
        'x_range': [-40,40],
        'y_range': [-40,40],
        'z_range': [-40,40],
        'X_range': [-40,40],
        'Y_range': [-40,40],
    },
    'midJuly': {
        'x_range': [-50,50],
        'y_range': [-50,50],
        'z_range': [-50,50],
        'X_range': [-50,70],
        'Y_range': [-50,100],
    },
    'earlyOct': {
        'x_range': [-60,60],
        'y_range': [-60,60],
        'z_range': [-60,60],
        'X_range': [-60,60],
        'Y_range': [-70,50],
    }
}

timeline = 'earlyOct'

PATH_PREP = os.path.join(homedir, r'3models\Preprocessing', timeline)
weights_dir = os.path.join(homedir, r'3models\Prediction', timeline, 'weights')
out_dir = os.path.join(homedir, timeline)
with open(os.path.join(PATH_PREP, 'FIPS.txt'), 'r') as f:
    FIPS_total = eval(f.read())
with open(os.path.join(PATH_PREP, 'columns_ctg.txt'), 'r') as f:
    columns_ctg = eval(f.read())

hparam = {
    "history_size": 7,          # Size of history window
    "NUM_CELLS": 128,           # Number of cells in LSTM layer
    "lr": 0.001,                # Learning rate
    "dp_ctg": 0.2,              # Dropout rate(categorical inputs)
    "dp_ts" : 0.2,              # Dropout rate(timeseries inputs)
    "EPOCHS": 15                # Number of epochs for training
}
target_size = 14
target_idx = 1

with open('misc\po_code_state_map.json', 'r') as f:
    po_code_state_map = json.load(f)

name_to_fips = {}
for st in po_code_state_map:
    name_to_fips[st['state']] = st['fips']

# Partition of counties by census division
West = "Alaska,Arizona,California,Colorado,Hawaii,Idaho,Montana,Nevada,New Mexico,Oregon,Utah,Washington,Wyoming"
West = [name_to_fips[name] for name in West.split(',')]
Midwest = "Illinois,Indiana,Iowa,Kansas,Michigan,Minnesota,Missouri,Nebraska,North Dakota,Ohio,South Dakota,Wisconsin"
Midwest = [name_to_fips[name] for name in Midwest.split(',')]
Northeast = "Connecticut,Maine,Massachusetts,New Hampshire,New Jersey,New York,Pennsylvania,Rhode Island,Vermont,District of Columbia"
Northeast = [name_to_fips[name] for name in Northeast.split(',')]
South = "Alabama,Arkansas,Delaware,Florida,Georgia,Kentucky,Louisiana,Maryland,Mississippi,North Carolina,Oklahoma,South Carolina,Tennessee,Texas,Virginia,West Virginia"
South = [name_to_fips[name] for name in South.split(',')]

_ = set()
for st in name_to_fips.values():
    _.add(st)

# colormap = dict(zip(sorted(_), range(len(_))))    # Partition by states (TOO many)
colormap = {}
for fips in _:
    if fips in West:
        colormap[fips] = 0
    elif fips in Midwest:
        colormap[fips] = 1
    elif fips in Northeast:
        colormap[fips] = 2
    elif fips in South:
        colormap[fips] = 3

x_range = config[timeline]['x_range']
y_range = config[timeline]['y_range']
z_range = config[timeline]['z_range']
X_range = config[timeline]['X_range']
Y_range = config[timeline]['Y_range']

# Load data and saved model
data_ctg = np.load(os.path.join(PATH_PREP, 'data_ctg.npy'), allow_pickle=True)
data_ts = np.load(os.path.join(PATH_PREP, 'data_ts.npy'), allow_pickle=True)

X_train, C_train, y_train = train_full(data_ts, data_ctg, target_idx, hparam["history_size"], target_size)

scaler_ts, scaler_ctg = get_StandardScaler(X_train, C_train)
mu, sigma = scaler_ts.mean_[target_idx], scaler_ts.scale_[target_idx]

X_train = normalizer(scaler_ts, X_train)
C_train = normalizer(scaler_ctg, C_train)

data_ctg = normalizer(scaler_ctg, data_ctg) # Normalize categorical values for later use

model = SingleLayerConditionalRNN(hparam["NUM_CELLS"], target_size,
                                categorical_dropout=hparam["dp_ctg"],
                                timeseries_dropout=hparam["dp_ts"],
                                cell="LSTM", sigma=sigma, mu=mu, ver='frozen')

train_data = load_Dataset(X_train, C_train, y_train)
for data in train_data:
    model(data[0])
    break
load_status = model.load_weights(weights_dir)
# load_status.assert_consumed() # Raise a weird assertion error stating the failure to load optimizer weights
load_status.assert_existing_objects_matched()

W = model.layer1.get_weights()[0]   # kernel weight
b = model.layer1.get_weights()[1]   # bias
initial_states = {}
for i, county in enumerate(data_ctg):
    initial_states[FIPS_total[i]] = np.matmul(county, W) + b

os.makedirs(out_dir, mode=0o770, exist_ok=True)
with open(os.path.join(out_dir, 'points.json'), 'w') as f:
    json.dump({key: initial_states[key].tolist() for key in initial_states}, f, separators=(',', ':'))

pca = PCA(n_components=3, svd_solver="full")
X = np.array(list(initial_states.values()))
pca.fit(X)

xyz, c, d, e, size = [], [], [], [], []
for county, pt in initial_states.items():
    xyz.append(pca.transform(pt.reshape(1,-1))[0])
    c.append(np.log(demo.loc[county, '#EligibleforMedicare2018']))
    d.append(np.log(1+demo.loc[county, 'mortality2015-17Estimated']))
    e.append(np.log(gdp.loc[county, '2015']))
    size.append(np.sqrt(demo.loc[county, 'PopulationDensityperSqMile2010']))
xyz = np.asarray(xyz)
df = pd.DataFrame(xyz, columns=['x','y','z'])
x_mean = df['x'].mean(); y_mean = df['y'].mean(); z_mean = df['z'].mean()
df['medicare'] = c
df['mortality'] = d
df['gdp'] = e
df['size'] = size
df.index = FIPS_total

fig = px.scatter_3d(df, x='x', y='y', z='z', opacity=0.7, color='medicare', size='size', hover_data=[df.index])
fig.update_traces(marker_sizemin=1)
fig.update_layout(scene_camera_center={'x':x_mean, 'y':y_mean, 'z':z_mean}, scene_xaxis_range=x_range, scene_yaxis_range=y_range, scene_zaxis_range=z_range, title_text=f'Color: log #medicare, size: sqrt population density')
fig.write_html(os.path.join(out_dir, '3D_medicare.html'))

fig = px.scatter_3d(df, x='x', y='y', z='z', opacity=0.7, color='mortality', size='size', hover_data=[df.index])
fig.update_traces(marker_sizemin=1)
fig.update_layout(scene_camera_center={'x':x_mean, 'y':y_mean, 'z':z_mean}, scene_xaxis_range=x_range, scene_yaxis_range=y_range, scene_zaxis_range=z_range, title_text=f'Color: log mortality, size: sqrt population density')
fig.write_html(os.path.join(out_dir, '3D_mortality.html'))

fig = px.scatter_3d(df, x='x', y='y', z='z', opacity=0.7, color='gdp', size='size', hover_data=[df.index])
fig.update_traces(marker_sizemin=1)
fig.update_layout(scene_camera_center={'x':x_mean, 'y':y_mean, 'z':z_mean}, scene_xaxis_range=x_range, scene_yaxis_range=y_range, scene_zaxis_range=z_range, title_text=f'Color: log gdp, size: sqrt population density')
fig.write_html(os.path.join(out_dir, '3D_gdp.html'))

pca = PCA(n_components=2, svd_solver="full")
X = np.array(list(initial_states.values()))
pca.fit(X)

xyz = []
for county, pt in initial_states.items():
    xyz.append(pca.transform(pt.reshape(1,-1))[0])
xyz = np.asarray(xyz)

df = pd.DataFrame(xyz, columns=['x','y'])
x_mean = df['x'].mean(); y_mean = df['y'].mean()
df['medicare'] = c
df['mortality'] = d
df['gdp'] = e
df['size'] = size
df.index = FIPS_total
fig = px.scatter(df, x='x', y='y', opacity=0.7, color='medicare', size='size', color_continuous_scale='viridis', hover_data=[df.index])
fig.update_traces(marker_sizemin=0.5)
fig.update_layout(xaxis_range=X_range, yaxis_range=Y_range, title_text=f'Color: log #medicare, size: sqrt population density')
fig.write_html(os.path.join(out_dir, '2D_medicare.html'))

fig = px.scatter(df, x='x', y='y', opacity=0.7, color='mortality', size='size', color_continuous_scale='viridis', hover_data=[df.index])
fig.update_traces(marker_sizemin=0.5)
fig.update_layout(xaxis_range=X_range, yaxis_range=Y_range, title_text=f'Color: log mortality, size: sqrt population density')
fig.write_html(os.path.join(out_dir, '2D_mortality.html'))

fig = px.scatter(df, x='x', y='y', opacity=0.7, color='gdp', size='size', color_continuous_scale='viridis', hover_data=[df.index])
fig.update_traces(marker_sizemin=0.5)
fig.update_layout(xaxis_range=X_range, yaxis_range=Y_range, title_text=f'Color: log gdp, size: sqrt population density')
fig.write_html(os.path.join(out_dir, '2D_gdp.html'))