"""Reproducible sentiment/QA experiment. Private text and outputs stay local."""
import argparse
import gc
import hashlib
import json
import random
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

STAGES = ['prepare', 'download', 'train', 'compare', 'select', 'evaluate', 'encode', 'features']
NAMES = ['negative', 'neutral', 'positive']


def absolute(value):
    p = Path(value)
    if not p.is_absolute(): raise ValueError(f'Absolute path required: {value}')
    return p


def sha(path):
    p = Path(path)
    if p.is_dir():
        return hashlib.sha256(''.join(f.relative_to(p).as_posix()+sha(f) for f in sorted(p.rglob('*')) if f.is_file() and '.cache' not in f.parts).encode()).hexdigest()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write_json(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def out(c): return absolute(c['work_dir'])


def device(c):
    import torch
    return ('cuda' if torch.cuda.is_available() else 'cpu') if c['device']=='auto' else c['device']


def receipt(c, stage, artifacts=None, details=None):
    p = out(c)/f'{stage}.receipt.json'
    if artifacts is not None:
        write_json(p, {'stage':stage, 'artifacts':{str(f.relative_to(out(c))):sha(f) for f in artifacts}, 'details':details or {}})
    if not p.exists(): raise ValueError(f'Run {stage} first')
    r = read_json(p)
    for name, digest in r['artifacts'].items():
        if not (out(c)/name).exists() or sha(out(c)/name)!=digest:
            raise ValueError(f'Stale/tampered artifact: {stage}/{name}; use a new work_dir')
    return r


def labels(c):
    p=absolute(c['labels']); raw=pd.read_excel(p) if p.suffix.lower()=='.xlsx' else pd.read_csv(p)
    d=raw[['text','label']].dropna().copy(); d['label']=pd.to_numeric(d.label, errors='raise')
    if not set(d.label.unique()) <= {-1,0,1}: raise ValueError('Expected human labels -1,0,1')
    d['text']=d.text.astype(str)
    normalized=d.text.map(lambda s: ''.join(unicodedata.normalize('NFKC',s).split()).lower())
    d['key']=normalized.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    d=d[normalized.ne('')].copy(); bad=d.groupby('key').label.nunique()
    d=d[~d.key.isin(bad[bad.gt(1)].index)].drop_duplicates('key').sort_values('key').reset_index(drop=True)
    d['label']=d.label.astype(int)+1
    if set(d.label)!=set(range(3)) or d.label.value_counts().min()<10:
        raise ValueError('Need at least 10 unique, non-conflicting annotations per class')
    return d


def prepare(c):
    from sklearn.model_selection import train_test_split
    d=labels(c); train, rest=train_test_split(d,test_size=.4,stratify=d.label,random_state=c['seed'])
    valid, test=train_test_split(rest,test_size=.5,stratify=rest.label,random_state=c['seed']); files=[]
    for name,frame in [('train',train),('validation',valid),('test',test)]:
        p=out(c)/f'{name}.csv';frame.to_csv(p,index=False,encoding='utf-8-sig');files.append(p)
    p=out(c)/'protocol.json'
    write_json(p, {'config':c,'labels_sha256':sha(c['labels']), 'rows_after_cleaning':len(d),
        'split_counts':dict(train=len(train),validation=len(valid),test=len(test)),
        'cleaning':'NFKC/whitespace/lower for grouping only; exclude conflicts; deduplicate before stratified 60/20/20',
        'test_use':'selected model evaluated once; no reselection from test'})
    receipt(c,'prepare',files+[p])


def download(c):
    from huggingface_hub import HfApi, snapshot_download
    lock={}
    for name,entry in c['models'].items():
        if 'local_path' in entry:
            dest=absolute(entry['local_path']); revision='local-'+sha(dest)
        else:
            revision=entry.get('revision') or HfApi().model_info(entry['repo']).sha
            dest=out(c)/'models'/name
            snapshot_download(entry['repo'],revision=revision,local_dir=str(dest))
        lock[name]={'repo':entry.get('repo'),'revision':revision,'path':str(dest),'sha256':sha(dest)}
    p=out(c)/'models.lock.json';write_json(p,lock);receipt(c,'download',[p])


def check_model(entry):
    if sha(entry['path'])!=entry['sha256']: raise ValueError('Downloaded checkpoint changed; use a new work_dir')


def train(c):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(c['seed'])
    entry=read_json(out(c)/'models.lock.json')['macbert'];check_model(entry)
    tok=AutoTokenizer.from_pretrained(entry['path'],local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(entry['path'],num_labels=3,local_files_only=True).to(device(c))
    model.config.id2label=dict(enumerate(NAMES));model.config.label2id={v:k for k,v in enumerate(NAMES)}
    d=pd.read_csv(out(c)/'train.csv')
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    rng=np.random.default_rng(c['seed']);losses=[];model.train()
    for epoch in range(c['epochs']):
        ids=rng.permutation(len(d));total=0
        for start in range(0,len(ids),c['batch_size']):
            b=d.iloc[ids[start:start+c['batch_size']]]
            x=tok(b.text.tolist(),padding=True,truncation=True,max_length=c['max_length'],return_tensors='pt').to(device(c))
            y=torch.tensor(b.label.to_numpy(),dtype=torch.long,device=device(c));optimizer.zero_grad(set_to_none=True)
            loss=model(**x,labels=y).loss;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();total+=loss.item()*len(b)
        losses.append(total/len(d));print(f'epoch {epoch+1}: loss={losses[-1]:.6f}',flush=True)
    p=out(c)/'macbert_finetuned';model.save_pretrained(p,safe_serialization=True);tok.save_pretrained(p)
    receipt(c,'train',[p],{'loss':losses,'train_rows':len(d),'device':device(c),'fixed_epochs':c['epochs']})
    del model,optimizer;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def score(c, path, texts, mapping=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).to(device(c)).eval()
    mapping=mapping or {str(k):str(v).lower() for k,v in model.config.id2label.items()}
    if model.config.num_labels!=3 or set(mapping.keys())!={'0','1','2'} or set(mapping.values())!=set(NAMES):
        raise ValueError(f'Verified three-class label mapping required for {path}; fill label_maps from model documentation')
    order=[NAMES.index(mapping[str(i)]) for i in range(3)];result=[]
    with torch.inference_mode():
        for i in range(0,len(texts),c['batch_size']):
            x=tok(texts[i:i+c['batch_size']],padding=True,truncation=True,max_length=c['max_length'],return_tensors='pt').to(device(c))
            probs=model(**x).logits.softmax(-1).cpu().numpy();aligned=np.empty_like(probs);aligned[:,order]=probs;result.append(aligned)
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return np.concatenate(result) if result else np.empty((0,3))


def metrics(y, probs):
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, recall_score
    p=probs.argmax(1)
    return {'n':len(y),'accuracy':accuracy_score(y,p),'macro_f1':f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0),
        'balanced_accuracy':balanced_accuracy_score(y,p),'recall_by_class':recall_score(y,p,labels=[0,1,2],average=None,zero_division=0).tolist(),
        'confusion_matrix':confusion_matrix(y,p,labels=[0,1,2]).tolist()}


def model_path(c,name):
    if name=='macbert':return str(out(c)/'macbert_finetuned')
    entry=read_json(out(c)/'models.lock.json')[name];check_model(entry);return entry['path']


def compare(c):
    d=pd.read_csv(out(c)/'validation.csv');rows=[]
    for name in c['models']:
        probs=score(c,model_path(c,name),d.text.tolist(),c['label_maps'].get(name));rows.append({'model':name,**metrics(d.label,probs)})
    rows.sort(key=lambda x:(-x['macro_f1'],x['model']))
    p=out(c)/'validation_comparison.json';write_json(p,rows);receipt(c,'compare',[p])


def select(c):
    rows=read_json(out(c)/'validation_comparison.json');winner=rows[0]['model'];mode=c['selection_mode']
    if mode not in ['historical_macbert','validation_best']:raise ValueError('Unknown selection_mode')
    selected='macbert' if mode=='historical_macbert' else winner;p=out(c)/'selection.json'
    write_json(p,{'selected':selected,'validation_winner':winner,'mode':mode,
        'reason':'historical pipeline reconstruction; not a new claim of superiority' if mode=='historical_macbert' else 'maximum validation macro-F1',
        'unequal_adaptation':'MacBERT fine-tuned on local training set; others use pretrained sentiment heads',
        'checkpoint_sha256':sha(model_path(c,selected))})
    receipt(c,'select',[p])


def evaluate(c):
    selection=read_json(out(c)/'selection.json');name=selection['selected'];d=pd.read_csv(out(c)/'test.csv')
    probs=score(c,model_path(c,name),d.text.tolist(),c['label_maps'].get(name))
    p=out(c)/'test_metrics.json';write_json(p,{'model':name,**metrics(d.label,probs),'use':'report only; never used for selection'});receipt(c,'evaluate',[p])


def read_qa(value):
    p=absolute(value);files=sorted(p.glob('*.csv')) if p.is_dir() else [p]
    if not files:raise ValueError('Provide private QA CSV(s); text is not shipped in this package')
    d=pd.concat([pd.read_csv(f,dtype=str,keep_default_na=False) for f in files],ignore_index=True)
    needed={'stock_code','datetime','question','reply_time','reply'}
    if not needed<=set(d):raise ValueError(f'Missing QA fields: {needed-set(d)}')
    if d.empty:raise ValueError('QA is empty')
    if not d.stock_code.str.fullmatch(r'\d{6}').all():raise ValueError('Six-digit stock_code required')
    qtime=pd.to_datetime(d.datetime,errors='coerce',format='mixed');atime=pd.to_datetime(d.reply_time,errors='coerce',format='mixed');answered=d.reply.str.strip().ne('')
    if qtime.isna().any() or (answered & (atime.isna() | atime.lt(qtime))).any():raise ValueError('Invalid question/reply timestamps')
    if 'whether_answer' in d and (answered & d.whether_answer.str.strip().str.lower().isin(['0','false','否'])).any():raise ValueError('Reply text conflicts with whether_answer')
    d['qa_id']=[hashlib.sha256((str(i)+'\0'+'\0'.join(row)).encode()).hexdigest() for i,row in enumerate(d[['stock_code','datetime','question','reply_time','reply']].itertuples(index=False,name=None))]
    return d


def encode(c):
    import torch
    from transformers import AutoTokenizer, AutoModel
    name=read_json(out(c)/'selection.json')['selected'];path=model_path(c,name);qa=read_qa(c['qa'])
    if c.get('expected_companies') is not None and qa.stock_code.nunique()!=c['expected_companies']:raise ValueError('QA company count differs from expected_companies')
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True);model=AutoModel.from_pretrained(path,local_files_only=True).to(device(c)).eval();artifacts=[]
    for side in ['question','reply']:
        p=out(c)/f'{side}.npy';artifacts.append(p);vectors=np.lib.format.open_memmap(p,mode='w+',dtype='float32',shape=(len(qa),model.config.hidden_size))
        with torch.inference_mode():
            for i in range(0,len(qa),c['batch_size']):
                texts=qa[side].iloc[i:i+c['batch_size']].tolist();x=tok(texts,padding=True,truncation=True,max_length=c['max_length'],return_tensors='pt').to(device(c))
                h=model(**x).last_hidden_state;mask=x.attention_mask.unsqueeze(-1)
                v=torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1).clamp_min(1),dim=1).cpu().numpy()
                v[np.array([not t.strip() for t in texts])]=0;vectors[i:i+len(v)]=v
        vectors.flush();del vectors
    p=out(c)/'embedding_manifest.json';artifacts.append(p)
    write_json(p,{'qa_sha256':sha(c['qa']),'qa_ids':qa.qa_id.tolist(),'rows':len(qa),'model':name,'checkpoint_sha256':sha(path),
        'max_length':c['max_length'],'pooling':'attention-mask mean including special tokens; L2; empty text zero'})
    receipt(c,'encode',artifacts);del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def scalar_features(c,qa,q,a):
    if q.shape!=a.shape or q.shape[0]!=len(qa) or not np.isfinite(q).all() or not np.isfinite(a).all():raise ValueError('Invalid vectors')
    d=qa[['qa_id','stock_code','datetime','reply_time']].copy();validq=qa.question.str.strip().ne('');answered=qa.reply.str.strip().ne('')
    d['question_length']=qa.question.str.len();d['reply_length']=qa.reply.str.len().where(answered)
    d['reply_lag_days']=(pd.to_datetime(qa.reply_time,errors='coerce',format='mixed')-pd.to_datetime(qa.datetime,format='mixed')).dt.total_seconds()/86400
    d['relevance']=np.clip((np.einsum('ij,ij->i',q,a)+1)/2,0,1);d.loc[~(answered & validq),'relevance']=np.nan;d.loc[~answered,'reply_lag_days']=np.nan
    d['template_ratio']=qa.reply.map(lambda s:min(1,sum(s.count(p)*len(p) for p in c['template_phrases'])/max(1,len(s)))).where(answered)
    d['novelty']=np.nan;days=pd.to_datetime(qa.reply_time,errors='coerce',format='mixed').dt.normalize()
    for _,g in d[answered].groupby('stock_code'):
        total=np.zeros(a.shape[1],dtype=np.float64);n=0
        for _,day in g.groupby(days.loc[g.index],sort=True):
            ids=day.index.to_numpy()
            if n:d.loc[ids,'novelty']=1-np.clip(a[ids]@(total/n),-1,1)
            total+=a[ids].sum(0,dtype=np.float64);n+=len(ids)
    for side in ['question','reply']:
        for name,pattern in c.get('event_patterns',{}).items():d[f'{side}_event_{name}']=qa[side].str.contains(pattern,regex=True).astype(int)
    return d


def features(c):
    qa=read_qa(c['qa']);m=read_json(out(c)/'embedding_manifest.json')
    if sha(c['qa'])!=m['qa_sha256'] or qa.qa_id.tolist()!=m['qa_ids']:raise ValueError('QA changed after encoding')
    name=read_json(out(c)/'selection.json')['selected'];path=model_path(c,name)
    if sha(path)!=m['checkpoint_sha256']:raise ValueError('Checkpoint changed after encoding')
    d=scalar_features(c,qa,np.load(out(c)/'question.npy',mmap_mode='r'),np.load(out(c)/'reply.npy',mmap_mode='r'))
    for side in ['question','reply']:
        probs=score(c,path,qa[side].tolist(),c['label_maps'].get(name));d[f'{side}_sentiment']=probs[:,2]-probs[:,0];d.loc[qa[side].str.strip().eq(''),f'{side}_sentiment']=np.nan
    common=['qa_id','stock_code'];qcols=[x for x in d if x.startswith('question_')];acols=[x for x in d if x.startswith('reply_') and x!='reply_time']+['relevance','template_ratio','novelty']
    qf=d[common+['datetime']+qcols].rename(columns={'datetime':'available_at'})
    af=d.loc[qa.reply.str.strip().ne(''),common+['reply_time']+acols].rename(columns={'reply_time':'available_at'});files=[]
    for filename,frame in [('question_features.csv',qf),('reply_features.csv',af)]:
        p=out(c)/filename;frame.to_csv(p,index=False,encoding='utf-8-sig');files.append(p)
    counts=[]
    for side,frame in [('question',qf),('reply',af)]:
        g=frame.assign(date=pd.to_datetime(frame.available_at,format='mixed').dt.strftime('%Y-%m-%d')).groupby(['stock_code','date']).size().rename('count').reset_index();g['side']=side;counts.append(g)
    p=out(c)/'calendar_day_counts.csv';pd.concat(counts,ignore_index=True).to_csv(p,index=False,encoding='utf-8-sig');files.append(p)
    p=out(c)/'features_manifest.json';write_json(p,{'qa_sha256':sha(c['qa']),'rows':len(qa),'stocks':int(qa.stock_code.nunique()),
        'question_date_range':[str(pd.to_datetime(qa.datetime,format='mixed').min()),str(pd.to_datetime(qa.datetime,format='mixed').max())],
        'availability':'question at datetime; reply/interaction at reply_time; counts are calendar-day descriptive statistics, map to trading sessions before prediction',
        'novelty':'1 - mean cosine to same-company replies on strictly earlier calendar days; cold start NaN','event_patterns':'fixed dictionary matches, not full RDIGF Q12/A19 rules'})
    receipt(c,'features',files+[p])


def execute(c, stage):
    for key in ['work_dir','labels','qa']:absolute(c[key])
    out(c).mkdir(parents=True,exist_ok=True);proto=out(c)/'protocol.json'
    if proto.exists():
        old=read_json(proto)
        if old['config']!=c or old['labels_sha256']!=sha(c['labels']):raise ValueError('Config or labels changed; use a new work_dir')
    elif stage!='prepare':raise ValueError('Run prepare first')
    for previous in STAGES[:STAGES.index(stage)]:receipt(c,previous)
    if (out(c)/f'{stage}.receipt.json').exists():
        receipt(c,stage)
        if stage in ['encode','features'] and read_json(out(c)/'embedding_manifest.json')['qa_sha256']!=sha(c['qa']):raise ValueError('QA changed; use a new work_dir')
        print(f'{stage}: verified cache',flush=True);return
    if stage=='evaluate' and (out(c)/'test_metrics.json').exists():raise ValueError('Incomplete test evaluation already exists')
    globals()[stage](c);print(f'{stage}: complete',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=STAGES+['benchmark','all']);p.add_argument('--config',required=True);args=p.parse_args();c=read_json(absolute(args.config))
    stages=STAGES if args.stage=='all' else (STAGES[:6] if args.stage=='benchmark' else [args.stage])
    for stage in stages:execute(c,stage)


if __name__=='__main__':main()
