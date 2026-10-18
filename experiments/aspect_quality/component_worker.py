"""Run the existing NLI component behind the updated answer-read guard."""
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.training_isolation import install_answer_read_guard
denied=install_answer_read_guard()
sys.path.insert(0,str(ROOT/'experiments/query_quality'))
runpy.run_path(str(ROOT/'experiments/query_quality/predict_cli.py'),run_name='__main__')
assert len(denied)==1
