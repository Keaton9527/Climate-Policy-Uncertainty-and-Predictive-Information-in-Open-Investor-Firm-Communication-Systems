"""Offline integration test using actual tiny Transformer models, never research results."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import workflow as w


class WorkflowTests(unittest.TestCase):
    def test_end_to_end_and_guards(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);models={}
            vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','公','司','业','绩','增','长','下','降','保','持','稳','定','问','回','复','谢','您']
            for k,name in enumerate(['macbert','finbert','hw2942']):
                dest=root/name;dest.mkdir();(dest/'vocab.txt').write_text('\n'.join(vocab),encoding='utf-8')
                tok=BertTokenizerFast(vocab_file=str(dest/'vocab.txt'));tok.save_pretrained(dest)
                torch.manual_seed(k)
                config=BertConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,max_position_embeddings=64,num_labels=3,
                    id2label=dict(enumerate(w.NAMES)),label2id={v:i for i,v in enumerate(w.NAMES)})
                BertForSequenceClassification(config).save_pretrained(dest,safe_serialization=True);models[name]={'local_path':str(dest)}
            labelled=pd.DataFrame([{'text':f'公司业绩 {label} 样本 {i}','label':label} for label in [-1,0,1] for i in range(30)])
            labelled=pd.concat([labelled,labelled.iloc[:1],pd.DataFrame([{'text':'conflict','label':-1},{'text':'conflict','label':1}])],ignore_index=True)
            labelled.to_csv(root/'labels.csv',index=False)
            qa=pd.DataFrame({'stock_code':['000001']*4,'datetime':['2020-01-01 09:00']*4,'question':['业绩增长','业绩下降','公司问题','公司问题'],
                'reply_time':['2020-01-02 09:00','2020-01-02 11:00','2020-01-03 09:00',''],'reply':['感谢您','公司回复','业绩增长','']})
            qa.to_csv(root/'qa.csv',index=False)
            c={'labels':str(root/'labels.csv'),'qa':str(root/'qa.csv'),'work_dir':str(root/'work'),'seed':42,'device':'cpu','models':models,
                'max_length':32,'batch_size':16,'epochs':1,'learning_rate':2e-5,'weight_decay':.01,'label_maps':{},'selection_mode':'historical_macbert',
                'expected_companies':None,'template_phrases':['感谢您'],'event_patterns':{'performance':'业绩'}}
            for stage in w.STAGES:w.execute(c,stage)
            protocol=w.read_json(root/'work/protocol.json');self.assertEqual(protocol['rows_after_cleaning'],90)
            keys=[set(pd.read_csv(root/f'work/{s}.csv').key) for s in ['train','validation','test']]
            self.assertFalse(keys[0]&keys[1] or keys[0]&keys[2] or keys[1]&keys[2])
            answer=pd.read_csv(root/'work/reply_features.csv');question=pd.read_csv(root/'work/question_features.csv')
            self.assertEqual(len(question),4);self.assertEqual(len(answer),3)
            self.assertTrue(answer.novelty.iloc[:2].isna().all());self.assertTrue(np.isfinite(answer.novelty.iloc[2]))
            self.assertTrue(answer.available_at.str.startswith('2020-01-0').all());self.assertNotIn('relevance',question)
            vectors=np.load(root/'work/reply.npy');self.assertTrue(np.all(vectors[3]==0));np.testing.assert_allclose(np.linalg.norm(vectors[:3],axis=1),1,atol=1e-6)
            before=w.sha(root/'work/test_metrics.json');w.execute(c,'evaluate');self.assertEqual(before,w.sha(root/'work/test_metrics.json'))
            with self.assertRaises(ValueError):w.execute(dict(c,seed=7),'compare')
            # Historical selection and best validation may differ, and both are recorded.
            selection=w.read_json(root/'work/selection.json');self.assertEqual(selection['selected'],'macbert');self.assertIn('validation_winner',selection)
            (root/'work/test_metrics.json').write_text('{}')
            with self.assertRaises(ValueError):w.execute(c,'evaluate')
            qa.loc[0,'reply_time']='2019-01-01';qa.to_csv(root/'qa.csv',index=False)
            with self.assertRaises(ValueError):w.read_qa(root/'qa.csv')
            with self.assertRaises(ValueError):w.execute(c,'encode')

    def test_known_novelty_and_absolute_path(self):
        qa=pd.DataFrame({'qa_id':['a','b','c'],'stock_code':['000001']*3,'datetime':['2020-01-01']*3,
            'reply_time':['2020-01-02','2020-01-02','2020-01-03'],'question':['x']*3,'reply':['x']*3})
        vectors=np.array([[1.,0.],[0.,1.],[1.,0.]])
        d=w.scalar_features({'template_phrases':[]},qa,vectors,vectors)
        self.assertTrue(d.novelty.iloc[:2].isna().all());self.assertAlmostEqual(d.novelty.iloc[2],.5)
        with self.assertRaises(ValueError):w.absolute('relative.csv')


if __name__=='__main__':unittest.main()
