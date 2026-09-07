"""从 Kaggle 下载原始数据到 data/raw/。

用法：python scripts/download_data.py

数据集：https://www.kaggle.com/datasets/efehandanisman/skytrax-airline-reviews
公开数据集，**不需要 Kaggle API token**，匿名即可下载（约 23 MB）。

下载完会自动跑一次自检，确认文件正确。
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_RAW

DATASET = "efehandanisman/skytrax-airline-reviews"
EXPECTED_MD5 = "3f16effe39b235e2f098008c6ba37a28"
FILENAME = "capstone_airline_reviews3.xlsx"


def main() -> None:
    target = DATA_RAW / FILENAME

    if target.exists():
        print(f"✅ 数据已存在：{target}")
        print("   想重新下载的话，先把它删掉。")
    else:
        try:
            import kagglehub
        except ImportError:
            print("❌ 缺少 kagglehub，先装：pip install kagglehub")
            sys.exit(1)

        print(f"从 Kaggle 下载 {DATASET} ...")
        cache_dir = Path(kagglehub.dataset_download(DATASET))

        src = cache_dir / FILENAME
        if not src.exists():                       # 万一上游改了文件名
            candidates = list(cache_dir.glob("*.xlsx")) + list(cache_dir.glob("*.csv"))
            if not candidates:
                print(f"❌ 下载目录里没找到数据文件：{cache_dir}")
                sys.exit(1)
            src = candidates[0]
            print(f"⚠️  文件名与预期不同，使用 {src.name}")

        DATA_RAW.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, DATA_RAW / src.name)
        target = DATA_RAW / src.name
        print(f"✅ 已复制到 {target}")

    # 校验文件指纹
    import hashlib
    md5 = hashlib.md5(target.read_bytes()).hexdigest()
    if md5 == EXPECTED_MD5:
        print(f"✅ 文件指纹正确 ({md5[:8]}...)")
    else:
        print(f"⚠️  指纹不一致：{md5[:8]}... (预期 {EXPECTED_MD5[:8]}...)")
        print("   可能 Kaggle 更新了数据集版本，下面的自检会告诉你差异。")

    print("\n" + "=" * 50)
    subprocess.run([sys.executable,
                    str(Path(__file__).parent / "profile_data.py")])


if __name__ == "__main__":
    main()
