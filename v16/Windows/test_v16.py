import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
import torch
import protocol
import adapter
import data_policy


class ProtocolTests(unittest.TestCase):
    def test_grid(self):
        js=protocol.jobs()
        self.assertEqual(len(js),174)
        self.assertEqual(len({j['id'] for j in js}),174)
        self.assertEqual(len(protocol.select(js,'key')),15)
        self.assertEqual(len(protocol.select(js,'main')),54)
        s=protocol.settings()
        self.assertTrue(all(j['batch_size']==s['batch_size'] and j['accum']==s['accum'] for j in js))
        self.assertTrue(all(j['epochs']==5 and j['lmax']==10 and j['lr']==2e-5 for j in js))

    def test_legacy_split_keeps_original_overlap_duplicates_and_conflicts(self):
        import json,tempfile
        from unittest.mock import patch
        rows={
            'train':[('甲',[(0,1,'A')]),('一',[]),('二',[]),('三',[]),('四',[])],
            'dev':[('甲',[(0,1,'A')]),('乙',[(0,1,'A')]),(' 乙 ',[(1,2,'B')])],
            'test':[('甲',[(0,1,'A')]),('丙',[]),('丙',[(0,1,'C')])],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'data'/'law';folder.mkdir(parents=True)
            for split,values in rows.items():
                with (folder/(split+'.jsonl')).open('w',encoding='utf-8') as f:
                    for text,gold in values:
                        f.write(json.dumps({'text':text,'entities':gold},ensure_ascii=False)+'\n')
            j={'domain':'law','train_cap':3,'dev_cap':0,'test_cap':0,'seed':1,'frac':1.}
            with patch.object(data_policy,'ROOT',root):
                train,dev,test,type_rows,audit=data_policy.prepare(j)
        self.assertEqual([t for t,g in train],['二','四','甲'])
        self.assertEqual(dev,rows['dev'])
        self.assertEqual(test,rows['test'])
        self.assertEqual(type_rows,[rows['train'][i] for i in [0,2,4]]+rows['dev'])
        self.assertEqual(audit['dev']['kept'],3)
        self.assertEqual(audit['evaluation']['kept'],3)

    def test_fast_matches_reference(self):
        old=adapter.legacy
        onto={'attract':[], 'tmutex':[('A','B')], 'imutex':[], 'parent':None}
        for mode in ['none','implicit','explicit']:
            old.set_seed(42)
            model=old.OntoSpan(old.StubEncoder(12),['∅','A','B'],onto,mode=mode,lmax=3)
            fast=copy.deepcopy(model)
            adapter.enable_fast(fast)
            data=[('甲乙丙丁',[(0,1,'A'),(1,3,'B')]),('戊己',[(0,2,'B')])]
            model.eval();fast.eval()
            reps,offs=model.encoder([data[0][0]])
            a,sp=model.span_reps(reps[0],offs[0]);b,sp2=fast.span_reps(reps[0],offs[0])
            self.assertEqual(sp,sp2)
            torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
            oa=torch.optim.SGD(model.parameters(),lr=.001)
            ob=torch.optim.SGD(fast.parameters(),lr=.001)
            old.set_seed(123)
            la=old.run_epoch(model,data,oa,'cpu',bs=2)
            old.set_seed(123)
            lb=adapter.train_epoch(fast,data,ob,'cpu',bs=2)
            self.assertAlmostEqual(la,lb,places=5)
            for k,v in model.state_dict().items():
                torch.testing.assert_close(v,fast.state_dict()[k],rtol=2e-5,atol=2e-6)

    def test_accumulation_and_anchor_loss(self):
        old=adapter.legacy
        onto={'attract':[('A','B',.5)],'tmutex':[], 'imutex':[('甲','乙')], 'parent':'A'}
        old.set_seed(42)
        reference=old.OntoSpan(old.StubEncoder(12),['∅','A','B'],onto,mode='implicit',lmax=3)
        fast=copy.deepcopy(reference);adapter.enable_fast(fast)
        data=[('甲乙丙丁',[(0,2,'A')])]*9
        a=torch.optim.SGD(reference.parameters(),lr=.001);b=torch.optim.SGD(fast.parameters(),lr=.001)
        old.set_seed(123);la=old.run_epoch(reference,data,a,'cpu',bs=4,accum=2)
        old.set_seed(123);lb=adapter.train_epoch(fast,data,b,'cpu',bs=4,accum=2)
        self.assertAlmostEqual(la,lb,places=4)
        for k,v in reference.state_dict().items():
            torch.testing.assert_close(v,fast.state_dict()[k],rtol=2e-5,atol=2e-6)


if __name__=='__main__': unittest.main()
