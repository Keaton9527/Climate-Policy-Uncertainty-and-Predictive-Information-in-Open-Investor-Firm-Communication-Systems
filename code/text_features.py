"""Train/evaluate text classifiers and return record features in memory."""
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

NAMES = ['negative', 'neutral', 'positive']


def device(config):
    import torch
    value = config['text']['device']
    return ('cuda' if torch.cuda.is_available() else 'cpu') if value == 'auto' else value


def load_classifier(config, name):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    settings = config['text']
    source = (settings.get('checkpoint') or settings['base_model']) if name == 'macbert' else settings[name+'_model']
    kwargs = {'cache_dir':config['paths']['model_cache']}
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=3, **kwargs).to(device(config))
    mapping = settings['label_maps'][name]
    order = [NAMES.index(mapping[str(i)]) for i in range(3)]
    return tokenizer, model, order


def split_annotations(annotations, config):
    frame = annotations.dropna(subset=['text']).copy()
    if not frame.label.isin([-1,0,1]).all():
        raise ValueError('Annotation labels must be -1, 0, 1')
    limit = config['text']['per_class_limit']
    parts = []
    for label in [-1,0,1]:
        part = frame[frame.label.eq(label)]
        parts.append(part.sample(n=limit, random_state=config['seed']) if len(part) >= limit else part)
    sampled = pd.concat(parts)
    sampled['label'] = sampled.label.astype(int) + 1
    return train_test_split(sampled, test_size=config['text']['evaluation_fraction'], random_state=config['seed'], stratify=sampled.label)


def train_classifier(tokenizer, model, frame, config):
    import torch
    from transformers import Trainer, TrainingArguments
    settings = config['text']
    encoded = tokenizer(frame.text.tolist(), padding='max_length', truncation=True, max_length=settings['max_length'])
    class AnnotationDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(frame)
        def __getitem__(self, index):
            return {**{key:torch.tensor(value[index]) for key,value in encoded.items()}, 'labels':torch.tensor(int(frame.label.iloc[index]))}
    args = TrainingArguments(output_dir=config['paths']['model_cache'], num_train_epochs=settings['epochs'],
        per_device_train_batch_size=settings['train_batch_size'], gradient_accumulation_steps=settings['gradient_accumulation_steps'],
        learning_rate=settings['learning_rate'], weight_decay=settings['weight_decay'],
        seed=config['seed'], data_seed=config['seed'], use_cpu=device(config)=='cpu', fp16=device(config).startswith('cuda'),
        report_to=[], save_strategy='no', logging_strategy='no', eval_strategy='no', optim='adamw_torch')
    Trainer(model=model, args=args, train_dataset=AnnotationDataset()).train()
    return model.eval()


def infer(texts, classifier, config, embeddings=False):
    import torch
    tokenizer, model, order = classifier
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=config['text']['max_length'], return_tensors='pt').to(device(config))
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=embeddings)
        probabilities = output.logits.softmax(-1).cpu().numpy()
        aligned = np.empty_like(probabilities)
        aligned[:,order] = probabilities
        vectors = output.hidden_states[-1][:,0,:].cpu().numpy() if embeddings else None
    return aligned, vectors


def extract_records(qa, classifier, config):
    frame = qa.dropna(subset=['question','datetime']).reset_index(drop=True).copy()
    frame['question_length'] = frame.question.str.len()
    frame['reply_length'] = frame.reply.fillna('').str.len().where(frame.whether_answer.eq(1), 0)
    frame['time_diff'] = ((frame.reply_time-frame.datetime).dt.total_seconds()/86400).round(2).where(frame.whether_answer.eq(1))
    for column,value in {'q_sentiment_score':0.,'a_sentiment_score':-1.,'q_sentiment_label':0,'a_sentiment_label':-1,'similarity':0.}.items():
        frame[column] = value
    batch = config['text']['batch_size']
    for start in range(0,len(frame),batch):
        part = frame.iloc[start:start+batch]
        qp,qv = infer(part.question.tolist(),classifier,config,True)
        frame.loc[part.index,'q_sentiment_score'] = qp[:,2]-qp[:,0]
        frame.loc[part.index,'q_sentiment_label'] = qp.argmax(1)-1
        answered = part.whether_answer.eq(1) & part.reply.fillna('').str.strip().ne('')
        if answered.any():
            ap,av = infer(part.loc[answered,'reply'].tolist(),classifier,config,True)
            frame.loc[part.index[answered],'a_sentiment_score'] = ap[:,2]-ap[:,0]
            frame.loc[part.index[answered],'a_sentiment_label'] = ap.argmax(1)-1
            q = qv[answered.to_numpy()]
            frame.loc[part.index[answered],'similarity'] = np.einsum('ij,ij->i',q,av)/np.maximum(np.linalg.norm(q,axis=1)*np.linalg.norm(av,axis=1),1e-12)
    return frame


def run_text(qa, annotations, config):
    from transformers import set_seed
    set_seed(config['seed'])
    train, evaluation = split_annotations(annotations, config)
    tokenizer, model, order = load_classifier(config,'macbert')
    if not config['text'].get('checkpoint'):
        model = train_classifier(tokenizer, model, train, config)
    macbert = (tokenizer, model.eval(), order)
    rows = []
    for name in ['macbert','finbert','bert']:
        classifier = macbert if name == 'macbert' else load_classifier(config,name)
        classifier[1].eval()
        batch = config['text']['batch_size']
        values = np.concatenate([infer(evaluation.text.iloc[i:i+batch].tolist(),classifier,config)[0] for i in range(0,len(evaluation),batch)])
        predicted = np.where((values[:,2]>values[:,0]) & (values[:,2]>values[:,1]),2,np.where(values[:,0]>values[:,1],0,1))
        recall = recall_score(evaluation.label,predicted,labels=[0,1,2],average=None,zero_division=0)
        rows.append(dict(Model=name,Accuracy=accuracy_score(evaluation.label,predicted),PositiveRecall=recall[2],NeutralRecall=recall[1],NegativeRecall=recall[0]))
        if name != 'macbert':
            del classifier
            gc.collect()
    records = extract_records(qa,macbert,config)
    return records, pd.DataFrame(rows)
