#!/usr/bin/env bash
# 把 airlabel 仓库的最新代码同步成主仓库里的存档快照。
#
# 为什么用 `git ls-files` 而不是 rsync/cp -R：
#   只拿被 git 跟踪的文件，于是所有 .gitignore 的东西天然不会进来 ——
#   secrets.toml（连接串、口令、账号哈希）、outputs/（标注数据和密码）、
#   node_modules/（26 MB）、.venv/。靠 --exclude 手工列举迟早会漏一个，
#   而漏的那次就是密钥进 git 历史的那次。
#
# 用法：bash scripts/sync_annotation_snapshot.sh [airlabel 仓库路径]
set -euo pipefail

SRC="${1:-$HOME/Desktop/airlabel}"
DST="$(cd "$(dirname "$0")/.." && pwd)/annotation"

[ -d "$SRC/.git" ] || { echo "❌ $SRC 不是 git 仓库"; exit 1; }

README_BANNER="$DST/README.md"
BANNER=""
[ -f "$README_BANNER" ] && BANNER="$(sed -n '1,/^---$/p' "$README_BANNER")"

rm -rf "$DST"; mkdir -p "$DST"
( cd "$SRC" && git ls-files -z ) | while IFS= read -r -d '' f; do
  mkdir -p "$DST/$(dirname "$f")"
  cp "$SRC/$f" "$DST/$f"
done

# 把「这是快照」的横幅重新贴回 README 顶部
if [ -n "$BANNER" ] && ! grep -q "这是快照，不是开发目录" "$DST/README.md"; then
  printf '%s\n\n%s' "$BANNER" "$(cat "$DST/README.md")" > "$DST/README.md.tmp"
  mv "$DST/README.md.tmp" "$DST/README.md"
fi

echo "✅ 同步完成：$(find "$DST" -type f | wc -l | tr -d ' ') 个文件"
for pat in secrets.toml credentials node_modules outputs .venv; do
  n=$(find "$DST" -path "*$pat*" -not -name 'secrets.example.toml' | wc -l | tr -d ' ')
  [ "$n" -eq 0 ] || { echo "❌ 快照里混进了 $pat，已中止"; exit 1; }
done
echo "✅ 安全检查通过（无 secrets / 凭据 / 依赖目录）"
