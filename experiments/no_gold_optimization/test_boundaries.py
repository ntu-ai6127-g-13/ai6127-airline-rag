"""Small safeguards for the experimental data boundary and partial-label accounting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from common import BASE,ROOT,install_gold_guard,metrics
from run_monitor import Monitor
import evaluate_gold


class BoundaryChecks(unittest.TestCase):
    def test_missing_weak_targets_are_not_fabricated(self):
        reference=pd.DataFrame({'text':['a','b','c'],'aspect':['seat',None,'food'],
                                'subjectivity':['subjective','objective',None],
                                'polarity':['positive',None,None],'label':[None,None,None]})
        prediction=pd.DataFrame({'text':['a','b','c'],'aspect':['seat','other','food'],
                                 'subjectivity':['objective','objective','subjective'],
                                 'polarity':['NA','NA','negative'],'label':['objective','objective','negative']})
        result=metrics(reference,prediction)
        self.assertEqual(result['task_counts'],{'aspect':2,'subjectivity':2,'polarity':1,'label':2})
        self.assertEqual(result['polarity_f1'],0.)  # Gated NA on known positive is still an error.

    def test_existing_monitor_survives_rerun(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'monitor.json';content='{"status":"preserve"}';p.write_text(content)
            with self.assertRaises(RuntimeError):Monitor(Path(d))
            self.assertEqual(p.read_text(),content)

    def test_modified_source_rejected_before_gold_open(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);source=out/'source.py';source.write_text('changed')
            (out/'frozen_candidates.json').write_text(json.dumps({'candidates':{},'scripts':{str(source):'wrong-hash'}}))
            with patch.object(evaluate_gold.pd,'read_csv',side_effect=AssertionError('Gold read before integrity check')) as read:
                with self.assertRaisesRegex(AssertionError,'source changed'):evaluate_gold.main(out)
                read.assert_not_called()

    def test_gold_and_previous_tuned_weights_blocked(self):
        denied=install_gold_guard()
        for path in [BASE/'inputs/gold_1073.csv',ROOT/'outputs/c_tuning_20261018/best.json']:
            with self.assertRaises(PermissionError):path.read_bytes()
        self.assertEqual(len(denied),3)  # Startup probe plus two explicit checks.


if __name__=='__main__':unittest.main()
