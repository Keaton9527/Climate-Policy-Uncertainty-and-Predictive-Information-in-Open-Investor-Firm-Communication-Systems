"""Paired date-block comparisons of frozen forecasts; no model refitting."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

SETS=['tech','tech_question','tech_answer','tech_all']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def holm(p):
    p=np.asarray(p); order=np.argsort(p); out=np.empty(len(p)); out[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1))); return out

def run_source_comparisons(config, dirs):
    cfg = config['bootstrap']
    output = dirs['source']
    regimes_path = dirs['desc_interim'] / 'market_regimes.csv'
    regimes = pd.read_csv(regimes_path).set_index('Window')['Trend'].to_dict()
    frames=[]; hashes={}; keys=None
    for fs in SETS:
        paths=sorted((dirs['main_interim'] / 'predictions').glob(f'window_*_XGBoost_{fs}.parquet')); assert len(paths)==10,(fs,len(paths))
        df=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True).sort_values(['window_id','datetime','stock_code']).reset_index(drop=True)
        assert not df.duplicated(['window_id','datetime','stock_code']).any()
        k=df[['window_id','datetime','stock_code','y_true']]
        if keys is None: keys=k
        else: pd.testing.assert_frame_equal(keys,k)
        assert set(df.y_true.unique())<={0,1} and set(df.y_pred.unique())<={0,1}
        for p in paths: hashes[p.name]=sha(p)
        for (window,date),g in df.groupby(['window_id','datetime'],sort=True):
            y=g.y_true.to_numpy(); pred=g.y_pred.to_numpy()
            tp=float(((y==1)&(pred==1)).sum()); tn=float(((y==0)&(pred==0)).sum()); fp=float(((y==0)&(pred==1)).sum()); fn=float(((y==1)&(pred==0)).sum())
            den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); m=(tp*tn-fp*fn)/den if den else 0.
            frames.append(dict(window_id=int(window),datetime=str(pd.Timestamp(date).date()),feature_set=fs,MCC=m,n_firms=len(g)))
    daily=pd.DataFrame(frames); wide=daily.pivot(index=['window_id','datetime'],columns='feature_set',values='MCC')[SETS]
    assert len(wide)==1212 and len(keys)==73528
    rng=np.random.default_rng(cfg['seed']); B=cfg['repetitions']; L=cfg['block_length']; means=[]; boot=[]; wins=[]
    for w,g in wide.groupby(level=0,sort=True):
        a=g.to_numpy(); n=len(a); starts=rng.integers(0,n,size=(B,math.ceil(n/L))); ix=((starts[:,:,None]+np.arange(L))%n).reshape(B,-1)[:,:n]
        means.append(a.mean(axis=0)); boot.append(a[ix].mean(axis=1)); wins.append(int(w))
    means=np.asarray(means); boot=np.stack(boot,axis=1)
    labels=np.asarray([regimes[w] for w in wins]); rows=[]
    def add(name,estimate,draws,group,comparison):
        p=(1+np.count_nonzero(np.abs(draws-estimate)>=abs(estimate)))/(B+1)
        lo,hi=np.quantile(draws,[.025,.975]); rows.append(dict(test=name,group=group,comparison=comparison,estimate=estimate,ci_low=lo,ci_high=hi,p_value=p,repetitions=B))
    for group in ['Overall','Downtrend','Sideways','Uptrend']:
        mask=np.ones(len(wins),dtype=bool) if group=='Overall' else labels==group
        for right,tag in [(1,'A_minus_Q'),(0,'A_minus_technical')]:
            add(group+'_'+tag,float((means[mask,2]-means[mask,right]).mean()),(boot[:,mask,2]-boot[:,mask,right]).mean(axis=1),group,tag)
    down=labels=='Downtrend'; up=labels=='Uptrend'
    add('Downtrend_minus_Uptrend_reply_increment',float((means[down,2]-means[down,0]).mean()-(means[up,2]-means[up,0]).mean()),(boot[:,down,2]-boot[:,down,0]).mean(axis=1)-(boot[:,up,2]-boot[:,up,0]).mean(axis=1),'Downtrend_minus_Uptrend','difference_in_A_minus_technical')
    results=pd.DataFrame(rows); results['holm_p_all_9']=holm(results.p_value)
    summaries=pd.DataFrame(means,columns=SETS); summaries.insert(0,'window_id',wins); summaries['regime']=labels
    report={'status':'completed','input_sha256':hashes,'regimes_sha256':sha(regimes_path),'config':cfg,'prediction_rows_per_model':len(keys),'dates':len(wide),'windows':len(wins),'firms':int(keys.stock_code.nunique()),'mean_MCC':dict(zip(SETS,means.mean(axis=0).tolist())),'inference_scope':'Conditional on fitted models, thresholds and these ten fixed historical windows; within-window date-block sampling; no inference over newly drawn market episodes.','exploratory':True}
    output.mkdir(parents=True,exist_ok=True)
    results.to_csv(output/'table_A7_source_comparisons.csv',index=False); summaries.to_csv(output/'01_window_mcc.csv',index=False); daily.to_csv(output/'02_daily_mcc.csv',index=False)
    (output/'03_analysis_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(results.to_string(index=False)); print('Mean MCC',report['mean_MCC'])
