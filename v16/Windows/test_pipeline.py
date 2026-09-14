import ast
import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
import torch
import adapter
from protocol import ROOT,jobs,manifest,digest
from worker import train
from integrity import validate
from metrics import record,paired
from report import comparisons,refresh


class PipelineTests(unittest.TestCase):
    def test_provenance(self):
        from protocol import file_hash
        p=json.loads((ROOT/'SOURCE_PROVENANCE.json').read_text(encoding='utf-8'))
        for source,target in [('run_code/ontospan_experiment.py','vendor/ontospan_experiment.py'),
                              ('run_code/baselines.py','vendor/baselines.py'),('build_datasets.py','build_datasets.py')]:
            self.assertEqual(p['original_sha256'][source],file_hash(ROOT/target))

    def test_graph_weight_and_signs(self):
        j=next(j for j in jobs() if j['family']=='main' and j['domain']=='law' and j['mode']=='explicit')
        onto=adapter.legacy.load_ontology('law');types=adapter.legacy.build_types([],onto)
        def make(j):
            adapter.legacy.set_seed(42)
            return adapter.make_model(j,types,onto,adapter.legacy.StubEncoder(12))
        real=make(j)
        old=adapter.legacy.SignedGAT(12,depth=j['gnn_depth']);old.load_state_dict(real.gnn.state_dict())
        torch.testing.assert_close(real.type_representations(),old(real.type_emb.weight,real.pos_nbr,real.neg_nbr),rtol=0,atol=0)
        zero=make(dict(j,me_weight=0.));removed=make(dict(j,perturb='removed'))
        torch.testing.assert_close(zero.type_representations(),removed.type_representations(),rtol=0,atol=0)
        random=make(dict(j,perturb='random_sign'))
        r,a=adapter.edges(real),adapter.edges(random)
        self.assertEqual(set(map(tuple,r['positive']+r['negative'])),set(map(tuple,a['positive']+a['negative'])))
        self.assertEqual(len(r['negative']),len(a['negative']))
        self.assertNotEqual(r['negative'],a['negative'])

    def test_47_prespecified_comparisons(self):
        self.assertEqual(len(comparisons(jobs())),47)

    def test_paired_identity_and_mismatch(self):
        a=[record('甲',[(0,1,'A')],[(0,1,'A')]),record('乙',[(0,1,'A')],[])]
        s=paired([a]*3,[a]*3,100)
        self.assertEqual(s['delta'],0);self.assertEqual(s['ci95'],[0.,0.]);self.assertEqual(s['p_randomization'],1.)
        b=copy.deepcopy(a);b[0]['sample_id']='different'
        with self.assertRaises(ValueError):paired([a]*3,[b]*3,100)

    def test_paired_statistics_cluster_duplicate_texts_without_dropping_them(self):
        a=[record('甲',[(0,1,'A')],set()),record('甲',[(0,1,'A')],set()),record('乙',[],set())]
        b=[record('甲',[(0,1,'A')],{(0,1,'A')}),record('甲',[(0,1,'A')],set()),record('乙',[],set())]
        result=paired([a],[b],replicates=100)
        self.assertEqual(result['clusters'],2)
        self.assertEqual(result['evaluation_rows'],3)
        self.assertGreater(result['delta'],0)

    def test_pair_direction_and_seed_pairing(self):
        bad=[record(str(i),[(0,1,'A')],[]) for i in range(12)]
        good=[record(str(i),[(0,1,'A')],[(0,1,'A')]) for i in range(12)]
        a=paired([bad]*3,[good]*3,200)
        b=paired([good]*3,[bad]*3,200)
        self.assertEqual(a['delta'],1.);self.assertEqual(b['delta'],-1.)
        self.assertEqual(a['p_randomization'],b['p_randomization'])
        self.assertEqual(a['ci95'],[1.,1.])

    def test_partial_report_with_real_valid_artifact(self):
        from worker import dump
        m=manifest();j=next(j for j in m['jobs'] if j['family']=='main' and j['domain']=='law' and j['mode']=='none')
        rows=[('甲乙',[(0,1,'受害人')])]
        audit=dict(capped_count=1,train_hash=digest(rows),dev_hash=digest(rows),evaluation_hash=digest(rows))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);dump(p/'MANIFEST.json',m)
            train(j,p/'jobs'/j['id'],'cpu',lambda:adapter.legacy.StubEncoder(12),(rows,rows,rows,rows,audit))
            good,missing,invalid=refresh(m,p,with_stats=True)
            self.assertEqual(len(good),1);self.assertEqual(len(missing),173);self.assertFalse(invalid)

    def test_changed_config_rejected(self):
        j=jobs()[0]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'COMPLETE.json').write_text(json.dumps({'job_digest':digest(dict(j,batch_size=4)),'files':{}}),encoding='utf-8')
            self.assertIn('configuration',validate(Path(tmp),j)[1])

    def test_all_nine_model_paths_and_tamper(self):
        rows=[('甲乙丙丁',[(0,1,'受害人'),(2,3,'犯罪嫌疑人')]),('戊己庚辛',[(0,2,'犯罪嫌疑人')])]
        audit=dict(capped_count=2,train_hash=digest(rows),dev_hash=digest(rows),evaluation_hash=digest(rows))
        data=(rows,rows,rows,rows,audit)
        for mode in ['none','implicit','explicit','sgcn','sigat','biaffine','global_pointer','bert_crf','w2ner']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                j=dict(jobs()[0],mode=mode,domain='law',epochs=1,batch_size=2,accum=1,lmax=3,id='TEST_'+mode)
                result=train(j,tmp,'cpu',lambda:adapter.legacy.StubEncoder(12),data)
                validated,error=validate(Path(tmp),j)
                self.assertIsNone(error);self.assertEqual(result['metrics'],validated['metrics'])
                p=Path(tmp)/'evaluation.jsonl';p.write_text(p.read_text(encoding='utf-8')+'\n',encoding='utf-8')
                self.assertIsNotNone(validate(Path(tmp),j)[1])

    def test_report_does_not_pass_missing_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=manifest();p=Path(tmp)
            good,missing,bad=refresh(m,p,with_stats=True)
            self.assertFalse(good);self.assertEqual(len(missing),174);self.assertFalse(bad)
            self.assertTrue((p/'FULL_SUMMARY.zip').exists())

    def test_explicit_encoding(self):
        for path in list(ROOT.glob('*.py'))+list((ROOT/'vendor').glob('*.py')):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            for n in ast.walk(tree):
                if not isinstance(n,ast.Call):continue
                name=getattr(n.func,'attr',getattr(n.func,'id',''))
                if name in ('read_text','write_text'):
                    self.assertTrue(any(k.arg=='encoding' for k in n.keywords),(path,n.lineno))
                if name=='open':
                    # Only file open calls, not the OS/runtime binary APIs.
                    args=n.args
                    mode=args[1] if isinstance(n.func,ast.Name) and len(args)>1 else (args[0] if isinstance(n.func,ast.Attribute) and args else None)
                    binary=isinstance(mode,ast.Constant) and isinstance(mode.value,str) and 'b' in mode.value
                    if not binary:self.assertTrue(any(k.arg=='encoding' for k in n.keywords),(path,n.lineno))


if __name__=='__main__':unittest.main()
