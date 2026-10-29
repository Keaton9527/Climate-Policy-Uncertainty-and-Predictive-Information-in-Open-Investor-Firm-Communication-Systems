"""Read source datasets once, at the pipeline boundary."""
from glob import glob
from pathlib import Path
import pandas as pd


def read_market(value):
    files = sorted(glob(value))
    if not files:
        raise FileNotFoundError(value)
    frames = []
    for filename in files:
        frame = pd.read_excel(filename) if Path(filename).suffix.lower() == '.xlsx' else pd.read_csv(filename, dtype={'stock_code':str})
        if 'TradingDate' in frame:
            frame = frame.rename(columns={'TradingDate':'datetime','Symbol':'stock_code','OpenPrice':'open','ClosePrice':'close',
                'HighPrice':'high','LowPrice':'low','Volume':'volume','Amount':'amount','ChangeRatio':'pct_chg'})
            frame['datetime'] = pd.to_datetime(frame.datetime, format='mixed', errors='coerce')
            frame = frame.dropna(subset=['datetime']).copy()
            for name in ['open','close','high','low','volume','amount','pct_chg']:
                frame[name] = pd.to_numeric(frame[name], errors='raise')
            frame['change'] = frame.close - frame.close/(1+frame.pct_chg)
            frame['pct_chg'] *= 100
            frame['volume'] /= 100
            frame['amount'] /= 1000
        frames.append(frame)
    market = pd.concat(frames, ignore_index=True)
    market['stock_code'] = market.stock_code.astype(str).str.extract(r'(\d{1,6})',expand=False).str.zfill(6)
    market['datetime'] = pd.to_datetime(market.datetime)
    columns = ['stock_code','datetime','open','high','low','close','change','pct_chg','volume','amount']
    market = market[columns].drop_duplicates().sort_values(['stock_code','datetime'])
    if market.duplicated(['stock_code','datetime']).any():
        raise ValueError('Conflicting market rows for the same stock and date')
    return market.reset_index(drop=True)


def read_inputs(config):
    paths = config['paths']
    # Check sources before model downloads or training.
    for key in ['qa','annotations','nev_index','ccpu','epu','exposure','exposure_words']:
        if not Path(paths[key]).is_file():
            raise FileNotFoundError(f'Missing input {key}: {paths[key]}')
    market = read_market(paths['market'])
    qa = pd.read_csv(paths['qa'],dtype={'stock_code':str})
    qa['stock_code'] = qa.stock_code.str.zfill(6)
    for name in ['datetime','reply_time']:
        qa[name] = pd.to_datetime(qa[name],format='mixed',errors='coerce')
    if 'whether_answer' not in qa:
        qa['whether_answer'] = qa.reply.fillna('').str.strip().ne('').astype(int)
    else:
        flags = qa.whether_answer.astype(str).str.lower().str.strip().map({'已回复':1,'未回复':0,'1':1,'0':0,'1.0':1,'0.0':0,'true':1,'false':0})
        if flags.isna().any():
            raise ValueError('whether_answer must be 1/0 or 已回复/未回复')
        qa['whether_answer'] = flags.astype(int)
    annotations = pd.read_csv(paths['annotations']) if Path(paths['annotations']).suffix.lower()=='.csv' else pd.read_excel(paths['annotations'])
    inputs = {'market':market,'qa':qa,'annotations':annotations,
        'nev_index':pd.read_excel(paths['nev_index']),
        'ccpu':pd.read_excel(paths['ccpu'],sheet_name=config['ccpu']['sheet'])[['date','CCPU']],
        'epu':pd.read_excel(paths['epu'],sheet_name='EPU 2000 onwards')[['year','month','EPU']],
        'exposure':pd.read_excel(paths['exposure']), 'exposure_words':pd.read_excel(paths['exposure_words'])}
    inputs['nev_index']['Date'] = pd.to_datetime(inputs['nev_index']['Date'])
    inputs['ccpu']['date'] = pd.to_datetime(inputs['ccpu']['date'])
    return inputs
