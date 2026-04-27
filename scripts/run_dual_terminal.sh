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

# 清理可能残留的旧 session
tmux kill-session -t avm 2>/dev/null || true

echo "log:  $LOGFILE"
echo "mem:  $MEMFILE"
echo "sock: $MEMSOCK"

# 左侧上 pane: 对话终端
tmux new-session -d -s avm \
    -n chat "cd '$PROJECT_DIR' && AVM_MEMDUMP='$MEMFILE' AVM_MEMSOCK='$MEMSOCK' python main.py 2>'$LOGFILE'; echo; echo '=== 会话结束 ==='; read -p '按 Enter 关闭...'"

# 右侧 pane: 日志 tail
tmux split-window -h -t avm \
    "echo '=== 日志 (实时) ==='; echo; tail --retry -f '$LOGFILE' 2>/dev/null; read -p '按 Enter 关闭...'"

# 左侧下 pane: 交互式内存检查器
tmux select-pane -t avm:0.0
tmux split-window -v -t avm:0.0 \
    "while [ ! -S '$MEMSOCK' ]; do sleep 0.3; done; python '$PROJECT_DIR/memshell.py' '$MEMSOCK'; read -p '按 Enter 关闭...'"

tmux select-pane -t avm:0.0
tmux attach-session -t avm
