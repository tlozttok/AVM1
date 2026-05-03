#!/usr/bin/env bash
set -e

TS="$(date +%H%M%S)_$$"
LOGFILE="/tmp/avm_${TS}.log"
MEMFILE="/tmp/avm_${TS}_mem.txt"
MEMSOCK="/tmp/avm_${TS}_mem.sock"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cleanup() {
    tmux kill-session -t avm 2>/dev/null || true
    rm -f "$MEMSOCK"
}
trap cleanup EXIT

tmux kill-session -t avm 2>/dev/null || true

echo "log:  $LOGFILE"
echo "mem:  $MEMFILE"
echo "sock: $MEMSOCK"

# 左侧: Web 服务器 (聊天界面在浏览器)
tmux new-session -d -s avm \
    -n web "cd '$PROJECT_DIR' && AVM_MEMDUMP='$MEMFILE' AVM_MEMSOCK='$MEMSOCK' python -m web.server 2>'$LOGFILE'; echo; echo '=== 会话结束 ==='; read -p '按 Enter 关闭...'"

# 右侧上: 日志 tail
tmux split-window -h -t avm \
    "echo '=== 日志 (实时) ==='; echo; tail --retry -f '$LOGFILE' 2>/dev/null; read -p '按 Enter 关闭...'"

# 右侧下: 交互式内存检查器
tmux split-window -v -t avm:0.1 \
    "while [ ! -S '$MEMSOCK' ]; do sleep 0.3; done; python '$PROJECT_DIR/memshell.py' '$MEMSOCK'; read -p '按 Enter 关闭...'"

echo "打开浏览器: http://127.0.0.1:8080"

tmux select-pane -t avm:0.0
tmux attach-session -t avm
