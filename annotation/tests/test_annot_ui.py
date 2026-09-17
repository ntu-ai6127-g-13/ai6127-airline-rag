import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import bcrypt
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from annot.config import USERS
from annot.db import LocalStore


class StreamlitTests(unittest.TestCase):
    def test_login_authenticated_page_and_no_foreign_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            store=LocalStore(Path(directory)/'store')
            store.initialize([{'id':'E0001','text_en':'The seat was comfortable.',
                'text_zh':'座位很舒适。','risk_flag':'','seq':1}],[])
            store.upsert({'username':'guo','sentence_id':'E0001','subjectivity':'subjective',
                'polarity':'negative','aspect':'seat','note':'OTHER_USER_PRIVATE_NOTE','updated_at':'2026-09-12T12:00:00+00:00'})
            with patch.dict(os.environ,{'STORAGE':'local','ANNOT_LOCAL_DIR':str(store.directory),'ANNOT_CACHE_DIR':str(Path(directory)/'cache')}):
                at=AppTest.from_file(str(ROOT/'app/annotate.py'),default_timeout=15)
                hashed=bcrypt.hashpw(b'test-password',bcrypt.gensalt(rounds=4)).decode()
                at.secrets={'storage':'local','annotators':{u:{'pw_hash':hashed} for u in USERS}}
                at.run()
                self.assertFalse(at.exception)
                self.assertEqual(at.text_input[0].label,'密码')
                at.text_input[0].input('test-password')
                next(b for b in at.button if b.label=='登录并继续标注').click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state['user'],'zhang')
                components=at.get('component_instance')
                self.assertTrue(components)
                combined=' '.join(str(c.proto) for c in components)
                self.assertNotIn('OTHER_USER_PRIVATE_NOTE',combined)
                self.assertIn('E0001',combined)


if __name__=='__main__': unittest.main()
