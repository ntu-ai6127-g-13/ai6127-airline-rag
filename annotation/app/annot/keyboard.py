from pathlib import Path
from streamlit.components.v1 import declare_component

_component = declare_component('airline_annotation', path=str(Path(__file__).parent / 'frontend'))


def workbench(**args):
    return _component(mode='work', key='annotation_workbench', default=None, **args)


def session_bridge(token='', clear=False):
    return _component(mode='auth', token=token, clear=clear, key='annotation_session', default=None)
