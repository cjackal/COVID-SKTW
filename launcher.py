import os
import sys
from datetime import datetime
import json
import logging
from misc.utility import *
from DataCleaner import DataCleaner
from LSTM_trainer import LSTM_trainer
from Scrapper import Scrapper

homedir = get_homedir()

config_name = os.path.join(homedir, "config.json")
if len(sys.argv)==2:
    ver = str(sys.argv[1])
elif len(sys.argv)==1:
    ver = 'frozen'
else:
    raise RuntimeError(f"Incompatible arguments of length {len(sys.argv)} was passed.")

"""
Set up logger.
"""
os.makedirs(os.path.join(homedir, 'LSTM', 'log'), mode=0o770, exist_ok=True)
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
handler_s = logging.StreamHandler()
handler_f = logging.FileHandler(os.path.join(homedir, f'LSTM/log/{datetime.now().strftime("%Y%m%d-%H%M%S")}.log'))
formatter = logging.Formatter('{name} {asctime}: {message}', datefmt='%d %b, %Y %H:%M:%S', style='{')
handler_s.setFormatter(formatter)
handler_f.setFormatter(formatter)
logger.addHandler(handler_s)
logger.addHandler(handler_f)

with open(config_name, 'r') as f:
    config_dict = json.load(f)

os.makedirs(os.path.join(homedir, 'data'), mode=0o770, exist_ok=True)
now = datetime.utcnow()
logger.info("Update timeseries data from remote.")
with open(os.path.join(homedir, 'data/date.txt'), 'r') as f:
    scrap_date = f.read(8)
if now.strftime('%Y%m%d')!=scrap_date:
    logger.info("Data not up-to-date. Start updating data.")
    Scrapper()
tmp = str(round(1000*now.timestamp()))

PATH_SCR = os.path.join(homedir, "LSTM/preprocessing")

STOP_PREP = False
logger.info("Search existing preprocessed data.")
for root, dirs, files in os.walk(PATH_SCR):
    for name in files:
        if name=='config.json':
            with open(os.path.join(root,name), 'r') as f:
                config_temp = json.load(f)
            already_there = True
            for key in ["date_generated", "start_train", "end_train", "start_date", "end_date"]:
                try:
                    already_there = (config_dict[key]==config_temp[key]) and already_there
                except:
                    already_there = False
                    break
            if already_there:
                tmp = os.path.basename(root)
                STOP_PREP = True
                break
    if STOP_PREP: break

PATH_SCR = os.path.join(PATH_SCR, tmp)

if not STOP_PREP:
    logger.info(f"Create new preprocessed data in {tmp}.")
    os.makedirs(PATH_SCR, mode=0o770, exist_ok=True)
    DataCleaner(config_name, tmp, ver)

logger.info(f"Use preprocessed data in {tmp}")
LSTM_trainer(config_name, tmp, ver)