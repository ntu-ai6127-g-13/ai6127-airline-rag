"""把测试用的名册钉死在三个人。

名册现在从 secrets 读（部署时会变，眼下只有孙思洁一个人）。测试断言的是
「三人标注」那套逻辑——多数票、分歧判定、一致率，不该跟着部署配置漂。

两条 import 路径都要打：测试里有的写 `from annot.db import ...`，
有的写 `from app.annot.db import ...`。Python 把这两条当成**不同的模块**，
各自持有一份 USERS，只改一边另一边照样是旧的。

之所以是 clear()+update() 而不是重新赋值：各模块都 `from .config import USERS`，
拿的是同一个 dict 对象的引用；重新绑定 config.USERS 它们看不到，原地改才生效。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'app'))

TEST_ROSTER = {'zhang': '张毓轩', 'guo': '郭泰来', 'sun': '孙宇鑫'}

for path in ('annot.config', 'app.annot.config'):
    try:
        module = __import__(path, fromlist=['USERS'])
    except ImportError:
        continue
    module.USERS.clear()
    module.USERS.update(TEST_ROSTER)
