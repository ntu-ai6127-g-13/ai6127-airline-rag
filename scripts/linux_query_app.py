"""streamlit run scripts/linux_query_app.py --server.port 8517"""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linux_rag import configure_models, ROOT

configure_models()
runpy.run_path(str(ROOT / "app/streamlit_app.py"), run_name="__main__")
