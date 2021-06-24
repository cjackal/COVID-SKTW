import os
import sys
from datetime import datetime
import json
import logging
from src.misc.utility import get_homedir
from src.DataCleaner import DataCleaner
from src.LSTM_trainer import LSTM_trainer
from src.Scrapper import Scrapper

homedir = get_homedir()
datadir = os.path.join(homedir, 'data')

forcePrep = False
ver = 'frozen'

if __name__=="__main__":
    if len(sys.argv)==3:
        config_name = sys.argv[-2]
        ver = sys.argv[-1]
    elif len(sys.argv)==2:
        if sys.argv[1][-4:]=='json':
            config_name = sys.argv[1]
        else:
            ver = str(sys.argv[1])
            config_name = os.path.join(homedir, "config.json")
    elif len(sys.argv)==1:
        config_name = os.path.join(homedir, "config.json")
    else:
        raise RuntimeError(f"Incompatible arguments of length {len(sys.argv)} was passed.")

    PATH_SCR = os.path.join(homedir, 'tmp')
    PATH_PREP = os.path.join(PATH_SCR, 'preprocessing')
    os.makedirs(PATH_PREP, mode=0o770, exist_ok=True)

    """
    Set up logger.
    """
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    os.makedirs(os.path.join(PATH_SCR, 'log'), mode=0o770, exist_ok=True)
    handler_s = logging.StreamHandler()
    handler_f = logging.FileHandler(os.path.join(PATH_SCR, 'log',
                                f'{datetime.now().strftime("%Y%m%d-%H%M%S")}.log'))
    formatter = logging.Formatter('{asctime} {name}: {message}',
                                    datefmt='%d %b, %Y %H:%M:%S', style='{')
    handler_s.setFormatter(formatter)
    handler_f.setFormatter(formatter)
    logger.addHandler(handler_s)
    logger.addHandler(handler_f)

    with open(config_name, 'r') as f:
        config_dict = json.load(f)

    os.makedirs(datadir, mode=0o770, exist_ok=True)
    now = datetime.utcnow()
    logger.info("Update timeseries data from remote.")
    try:
        with open(os.path.join(datadir, 'date.txt'), 'r') as f:
            scrap_date = f.read(8)
    except:
        scrap_date = '0'*8
    if now.strftime('%Y%m%d')!=scrap_date:
        logger.info("Data not up-to-date. Start updating data.")
        Scrapper()
    tmp = str(round(1000*now.timestamp()))


    STOP_PREP = False
    if forcePrep:
        logger.info("Forcing preperation without using the existing files.")
    else:
        logger.info("Search existing preprocessed data.")
        for root, dirs, files in os.walk(PATH_PREP):
            for name in files:
                if name=='config.json':
                    with open(os.path.join(root, name), 'r') as f:
                        config_temp = json.load(f)
                    is_exist = True
                    for key in ["date_generated", "start_train", "end_train", "start_date", "end_date"]:
                        try:
                            is_exist &= (config_dict[key]==config_temp[key])
                        except:
                            is_exist = False
                            break
                    if is_exist:
                        tmp = os.path.basename(root)
                        STOP_PREP = True
                        break
            if STOP_PREP: break

    PATH_PREP = os.path.join(PATH_PREP, tmp)

    if not STOP_PREP:
        logger.info(f"Create new preprocessed data in {tmp}.")
        os.makedirs(PATH_SCR, mode=0o770, exist_ok=True)
        DataCleaner(config_name, tmp, ver)

    logger.info(f"Use preprocessed data in {tmp}")
    LSTM_trainer(config_name, tmp, ver)