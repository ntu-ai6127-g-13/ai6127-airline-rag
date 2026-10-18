from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments/query_quality"))
from prompts import parse_output


class QueryModelBoundaryTests(unittest.TestCase):
    def test_ambiguous_output_is_an_error_not_a_successful_prediction(self):
        result=parse_output("seat positive, food negative")
        self.assertFalse(result["valid"])
        self.assertEqual(result["aspect"],"invalid")
        self.assertFalse(parse_output("This is probably fine")["valid"])

    def test_objective_prediction_has_no_polarity(self):
        result=parse_output("value objective")
        self.assertTrue(result["valid"])
        self.assertEqual(result["polarity"],"NA")

    def test_guard_blocks_answers_in_a_child_process(self):
        script="""
import sys
from pathlib import Path
sys.path.insert(0,'experiments/query_quality')
from runtime import guard_gold,ROOT
denied=guard_gold()
paths=['data/eval/to_label.csv','outputs/c_no_gold_optimization_20261018/gold_audit.json','outputs/c_tuning_20261018/best.json']
for relative in paths:
 try: open(ROOT/relative)
 except PermissionError: pass
 else: raise AssertionError(relative)
assert len(denied)==4
print('boundary passed')
"""
        result=subprocess.run([sys.executable,"-c",script],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=="__main__":unittest.main()
