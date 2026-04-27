#!/usr/bin/env bash
set -e

LOGFILE="/tmp/avm_$(date +%H%M%S)_$$.log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cleanup() {
    tmux kill-session -t avm 2>/dev/null || true
}
trap cleanup EXIT

echo "log file: $LOGFILE"

# 用 tmux 开两个 pane：
#   左 pane: 干净对话终端 (stderr → logfile)
#   右 pane: 日志 tail
tmux new-session -d -s avm \
    -n chat "cd '$PROJECT_DIR' && python main.py 2>'$LOGFILE'; echo; echo '=== 会话结束 ==='; read -p '按 Enter 关闭...'"

tmux split-window -h -t avm \
    "echo '=== 日志 (实时) ==='; echo '文件: $LOGFILE'; echo; tail --retry -f '$LOGFILE' 2>/dev/null; read -p '按 Enter 关闭...'"

tmux select-pane -t avm:0.0
tmux attach-session -t avm
