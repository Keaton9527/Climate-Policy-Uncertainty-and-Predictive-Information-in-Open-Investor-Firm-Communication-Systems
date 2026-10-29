"""Section 4.4 statistics adapted from experiments 015, 016 and 017 (run 002)."""
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from climate import block_counts, holm, uncertainty

def weighted_auc(y, score, dateix, counts):
    """Pooled AUC with date multiplicities and exact half credit for tied scores."""
    order = np.argsort(score, kind='stable')
    y, score, dateix = y[order], score[order], dateix[order]
    starts = np.r_[0, np.flatnonzero(np.diff(score))+1]
    result = np.full(len(counts), np.nan)
    for lo in range(0, len(counts), 64):
        w = counts[lo:lo+64, dateix].astype(float)
        pos = np.add.reduceat(w*y, starts, axis=1)
        neg = np.add.reduceat(w*(1-y), starts, axis=1)
        u = (pos * (np.cumsum(neg,axis=1) - .5*neg)).sum(axis=1)
        denom = pos.sum(axis=1)*neg.sum(axis=1)
        result[lo:lo+64] = np.divide(u,denom,out=np.full(len(w),np.nan),where=denom>0)
    return result


def corr(a,b):
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:
        return np.nan
    return np.corrcoef(a,b)[0,1]


def metric_arrays(g, dates, counts, specs):
    di = dates.get_indexer(g.datetime)
    y = g.y_true.to_numpy()
    result, details = {}, {}
    daily = {spec: np.full((len(dates),2),np.nan) for spec in specs}
    for dt, gg in g.groupby('datetime', sort=True):
        k = dates.get_loc(dt)
        ret = gg.forward_return.to_numpy()
        for spec in specs:
            prob = gg[f'y_prob_{spec}'].to_numpy()
            daily[spec][k] = [corr(prob,ret),corr(rankdata(prob),rankdata(ret))]
    for spec in specs:
        pred = g[f'y_pred_{spec}'].to_numpy()
        prob = g[f'y_prob_{spec}'].to_numpy()
        conf = np.stack([np.bincount(di, weights=((y==a)&(pred==b)), minlength=len(dates)) for a,b in [(0,0),(0,1),(1,0),(1,1)]],axis=1)
        tn,fp,fn,tp = (counts@conf).T
        den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc = np.divide(tp*tn-fp*fn,den,out=np.zeros(len(counts)),where=den>0)
        mcc[(tn+fp+fn+tp)==0] = np.nan
        auc = weighted_auc(y,prob,di,counts)
        result[(spec,'MCC')], result[(spec,'AUC')] = mcc,auc
        if len(np.unique(y))==2:
            np.testing.assert_allclose(auc[0],roc_auc_score(y,prob),atol=1e-12)
        np.testing.assert_allclose(mcc[0],matthews_corrcoef(y,pred),atol=1e-12)
        details[spec] = dict(tn=conf[:,0].sum(),fp=conf[:,1].sum(),fn=conf[:,2].sum(),tp=conf[:,3].sum(),up_share=y.mean())
    for spec in specs[1:]:
        for metric in ['MCC','AUC']:
            result[(spec,'delta_'+metric)] = result[(spec,metric)]-result[('tech',metric)]
        for j,metric in enumerate(['IC','RankIC']):
            valid = np.isfinite(daily[spec][:,j]) & np.isfinite(daily['tech'][:,j])
            den = counts@valid.astype(float)
            for s in [spec,'tech']:
                num = counts@np.where(valid,daily[s][:,j],0.)
                result[(s,metric+'_'+spec)] = np.divide(num,den,out=np.full(len(counts),np.nan),where=den>0)
            result[(spec,'delta_'+metric)] = result[(spec,metric+'_'+spec)]-result[('tech',metric+'_'+spec)]
            details[spec]['valid_dates_'+metric] = int(valid.sum())
        values = g[f'd_{spec}'].to_numpy()
        num = counts@np.bincount(di,weights=values,minlength=len(dates))
        den = counts@np.bincount(di,minlength=len(dates))
        result[(spec,'delta_BrierImprovement')] = np.divide(num,den,out=np.full(len(counts),np.nan),where=den>0)
    return result, details


def grouped_analysis(panel, groupcol, statecol, config, specs, metrics):
    dates = pd.DatetimeIndex(sorted(panel.datetime.unique()))
    counts = block_counts(len(dates),config['repetitions'],config['block_length'],config['seed'])
    arrays,rows,coverage,contrasts = {},[],[],[]
    for (group,state),g in panel.groupby([groupcol,statecol],observed=True,sort=True):
        a,d = metric_arrays(g,dates,counts,specs)
        arrays[(group,state)] = a
        coverage.append(dict(group=group,state=state,n_obs=len(g),n_dates=g.datetime.nunique(),n_firms=g.stock_code.nunique(),n_windows=g.window_id.nunique(),windows=','.join(map(str,sorted(g.window_id.unique())))))
        for spec in specs[1:]:
            for metric in metrics:
                row = dict(specification=spec,metric=metric,**uncertainty(a[(spec,'delta_'+metric)]),**coverage[-1])
                if metric in ['MCC','AUC']:
                    row['baseline'] = a[('tech',metric)][0]
                    row['augmented'] = a[(spec,metric)][0]
                elif metric in ['IC','RankIC']:
                    row['baseline'] = a[('tech',metric+'_'+spec)][0]
                    row['augmented'] = a[(spec,metric+'_'+spec)][0]
                    row['valid_metric_dates'] = d[spec]['valid_dates_'+metric]
                rows.append(row)
        for spec in specs:
            coverage[-1].update({spec+'_'+k:v for k,v in d[spec].items()})
    for group in sorted(panel[groupcol].dropna().unique()):
        if (group,'low') not in arrays or (group,'high') not in arrays:continue
        for spec in specs[1:]:
            for metric in metrics:
                x = arrays[(group,'low')][spec,'delta_'+metric]-arrays[(group,'high')][spec,'delta_'+metric]
                contrasts.append(dict(group=group,specification=spec,metric=metric,contrast='low_minus_high',**uncertainty(x)))
    if {'high_exposure','low_exposure'}.issubset(set(panel[groupcol].unique())):
        for spec in specs[1:]:
            for metric in metrics:
                hi = arrays['high_exposure','low'][spec,'delta_'+metric]-arrays['high_exposure','high'][spec,'delta_'+metric]
                lo = arrays['low_exposure','low'][spec,'delta_'+metric]-arrays['low_exposure','high'][spec,'delta_'+metric]
                contrasts.append(dict(group='high_minus_low_exposure',specification=spec,metric=metric,contrast='signed_difference_in_differences',**uncertainty(hi-lo)))
    c = pd.DataFrame(contrasts, columns=['group','specification','metric','contrast','estimate','ci_low','ci_high','p_value','valid_repetitions'])
    c['holm_p_primary_family'] = np.nan
    mask = c.specification.eq('tech_answer') & c.metric.eq('MCC') & c.contrast.ne('absolute_gap_difference')
    c.loc[mask,'holm_p_primary_family'] = holm(c.loc[mask,'p_value'])
    return pd.DataFrame(rows), c


def controls(calendar, epu_lag, inputs):
    m = inputs['nev_index'].copy().rename(columns={'Date':'datetime'}).sort_values('datetime').set_index('datetime')
    returns = m.close.pct_change(fill_method=None)
    m['nev_rv'] = returns.rolling(20,min_periods=20).std(ddof=0)
    m['nev_ret20'] = m.close.pct_change(20,fill_method=None)
    ccpu = inputs['ccpu'].copy().set_index('date').CCPU
    e = inputs['epu'].copy()
    e.index = pd.PeriodIndex(pd.to_datetime(dict(year=e.year,month=e.month,day=1)),freq='M')
    f = pd.DataFrame({'datetime':calendar})
    f['ccpu'] = (f.datetime-pd.Timedelta(days=1)).map(ccpu)
    f['epu'] = (f.datetime.dt.to_period('M')-epu_lag).map(e.EPU)
    f = f.merge(m[['nev_rv','nev_ret20']],left_on='datetime',right_index=True,how='left',validate='one_to_one')
    f['trend'] = (f.datetime-pd.Timestamp('2017-01-01')).dt.days/365.25
    return f


def residualize(p,b,inputs,epu_lag):
    cal = pd.DatetimeIndex(sorted(b.datetime.unique()))
    f = controls(cal,epu_lag,inputs)
    windows = p[['window_id','train_start','train_end']].drop_duplicates().sort_values('window_id')
    coefficients, maps = [],[]
    cols = ['epu','nev_rv','nev_ret20','trend']
    for w in windows.itertuples(index=False):
        tr = f[f.datetime.between(pd.Timestamp(w.train_start),pd.Timestamp(w.train_end))].dropna()
        assert len(tr)>100
        mu,sd = tr[cols].mean(),tr[cols].std(ddof=0)
        x = np.c_[np.ones(len(tr)),(tr[cols]-mu)/sd]
        beta = np.linalg.lstsq(x,tr.ccpu,rcond=None)[0]
        resid = tr.ccpu-x@beta
        qlo,qhi = np.quantile(resid,[.25,.75])
        out = f[f.datetime.isin(p.loc[p.window_id.eq(w.window_id),'datetime'].unique())].copy()
        assert not out.isna().any().any()
        out['residual_ccpu'] = out.ccpu-np.c_[np.ones(len(out)),(out[cols]-mu)/sd]@beta
        out['residual_state'] = np.where(out.residual_ccpu<=qlo,'low',np.where(out.residual_ccpu>=qhi,'high','middle'))
        out['window_id'] = w.window_id
        maps.append(out.drop(columns='ccpu'))
        row = dict(window_id=w.window_id,epu_lag_months=epu_lag,training_dates=len(tr),train_start_available=tr.datetime.min(),train_end_available=tr.datetime.max(),requested_train_start=w.train_start,requested_train_end=w.train_end,q25=qlo,q75=qhi,r_squared=1-np.var(resid)/np.var(tr.ccpu),intercept=beta[0],rank=np.linalg.matrix_rank(x),condition_number=np.linalg.cond(x))
        for j,c in enumerate(cols):row.update({c+'_coef':beta[j+1],c+'_mean':mu[c],c+'_sd':sd[c]})
        coefficients.append(row)
    return p.merge(pd.concat(maps),on=['window_id','datetime'],validate='many_to_one'),pd.DataFrame(coefficients),f


def regress(panel,outcome,terms):
    """OLS with firm/year FE and Newey-West covariance of summed date scores.

    Date aggregation preserves arbitrary contemporaneous cross-sectional dependence;
    Bartlett HAC at 20 and 60 trading dates permits temporal dependence.
    """
    p = panel.dropna(subset=terms+[outcome]).sort_values(['datetime','stock_code']).copy()
    p['year'] = p.datetime.dt.year.astype(str)
    z = (p[terms]-p[terms].mean())/p[terms].std(ddof=0)
    scale = dict(zip(terms,p[terms].std(ddof=0)))
    # Interaction is product of standardized inputs, with both main effects included.
    if 'exposure' in terms:
        c = 'residual_ccpu' if 'residual_ccpu' in terms else 'ccpu'
        z[c+'_x_exposure'] = z[c]*z.exposure
    fe = pd.get_dummies(p[['stock_code','year']],drop_first=True,dtype=float)
    design = pd.concat([pd.Series(1.,index=p.index,name='Intercept'),z,fe],axis=1)
    x,y = design.to_numpy(float),p[outcome].to_numpy(float)
    assert np.linalg.matrix_rank(x)==x.shape[1]
    beta = np.linalg.lstsq(x,y,rcond=None)[0]
    residual = y-x@beta
    inv = np.linalg.inv(x.T@x)
    scores = pd.DataFrame(x*residual[:,None]).groupby(p.datetime.to_numpy(),sort=True).sum().to_numpy()
    rows = []
    for lag in [20,60]:
        meat = scores.T@scores
        for k in range(1,min(lag+1,len(scores))):
            cross = scores[k:].T@scores[:-k]
            meat += (1-k/(lag+1))*(cross+cross.T)
        cov = inv@meat@inv * len(p)/(len(p)-x.shape[1])
        se = np.sqrt(np.maximum(np.diag(cov),0))
        for j,term in enumerate(design.columns[:1+z.shape[1]]):
            rows.append(dict(term=term,estimate=beta[j],se=se[j],ci_low=beta[j]-1.96*se[j],ci_high=beta[j]+1.96*se[j],p_value=2*norm.sf(abs(beta[j]/se[j])),hac_lags=lag,n_obs=len(p),n_dates=p.datetime.nunique(),n_firms=p.stock_code.nunique(),raw_sd=scale.get(term,np.nan),r_squared=1-np.var(residual)/np.var(y)))
    return pd.DataFrame(rows)


def prepare_exposure(inputs):
    raw=inputs['exposure_words'].copy()
    raw['stock_code']=raw.文件名.str[:6]
    raw['report_year']=raw.文件名.str[7:11].astype(int)
    raw['publication_date']=pd.to_datetime(raw.文件名.str.extract(r'(\d{4}-\d{1,2}-\d{1,2})$',expand=False),format='mixed')
    assert not raw.duplicated(['stock_code','report_year']).any()
    final=inputs['exposure'].copy()
    final['stock_code']=final.stkcd.astype(str).str.zfill(6)
    final['report_year']=final.year
    assert not final.duplicated(['stock_code','report_year']).any()
    raw['转型_reconstructed']=raw.loc[:,'节能':'集约'].sum(axis=1)
    check=final.merge(raw,on=['stock_code','report_year'],how='left',validate='one_to_one')
    assert check.文件名.notna().all()
    audits=[]
    for label in ['转型']:
        error=check[label+'风险总词频']-check[label+'_reconstructed']
        assert (error==0).all()
        audits.append(dict(check=label+'_word_counts',n=len(check),mismatches=int((error!=0).sum())))
    assert (check.年报全文总词数==check.总词数).all()
    np.testing.assert_allclose(check.转型风险指数,100*check.转型_reconstructed/check.总词数,atol=1e-9)
    e=check[['stock_code','report_year','publication_date','转型风险指数']].rename(columns={'转型风险指数':'exposure'})
    e['available_date']=e.publication_date+pd.Timedelta(days=1)
    return e


def attach_actual_exposure(p,b,e):
    events=e.dropna(subset=['available_date']).copy()
    # A delayed/reissued older report must not replace a newer available fiscal year.
    events=events.sort_values(['stock_code','available_date','report_year'])
    events=events[events.report_year.eq(events.groupby('stock_code').report_year.cummax())]
    events=events.drop_duplicates(['stock_code','available_date'],keep='last')
    events=events[['stock_code','report_year','publication_date','available_date','exposure']].sort_values('available_date')
    def merge(f):
        return pd.merge_asof(f.sort_values('datetime'),events,left_on='datetime',right_on='available_date',by='stock_code',direction='backward',allow_exact_matches=True)
    q=merge(p)
    train=merge(b[['stock_code','datetime']])
    thresholds=[]
    for w in p[['window_id','train_start','train_end']].drop_duplicates().itertuples(index=False):
        tr=train[train.datetime.between(pd.Timestamp(w.train_start),pd.Timestamp(w.train_end))].dropna(subset=['exposure']).drop_duplicates(['stock_code','report_year'])
        thresholds.append(dict(window_id=w.window_id,median=tr.exposure.median(),training_firm_years=len(tr),training_firms=tr.stock_code.nunique()))
    th=pd.DataFrame(thresholds)
    q=q.merge(th[['window_id','median']],validate='many_to_one')
    q['exposure_group']=np.where(q.exposure>q['median'],'high_exposure','low_exposure')
    q.loc[q.exposure.isna(),'exposure_group']=None
    assert (q.loc[q.exposure.notna(),'publication_date']<q.loc[q.exposure.notna(),'datetime']).all()
    return q,th
