"""交互式内存检查器 — 通过 Unix socket 查询 Memory，类似 LLM 的 memory_read"""
import socket
import sys

HELP = """命令:
  <path>    查询路径，如 $MEM.calc.result1 或直接 system
  /h, /help 帮助
  /t, /tree 显示完整树
  /q, /quit 退出
"""


def _recv_full(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("server closed")
        buf += chunk
    return buf


def main():
    if len(sys.argv) < 2:
        print("用法: python memshell.py <socket_path>")
        sys.exit(1)

    sock_path = sys.argv[1]
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    try:
        sock.connect(sock_path)
    except (FileNotFoundError, ConnectionRefusedError):
        print("无法连接到内存服务，等待...", flush=True)
        sock.close()
        return

    print("=== 内存检查器 ===")
    print("输入路径 (如 $MEM.calc) 查询 /h 帮助 /q 退出")
    print()

    while True:
        try:
            line = input("mem> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in ("/q", "/quit", "/exit"):
            break
        if line in ("/h", "/help"):
            print(HELP, end="")
            continue
        if line in ("/t", "/tree"):
            line = "$MEM"

        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(sock_path)
            sock.sendall(line.encode("utf-8"))

            header = _recv_full(sock, 8)
            length = int(header.decode(), 16)
            resp = _recv_full(sock, length).decode("utf-8")

            print(resp)
        except (BrokenPipeError, ConnectionResetError, ConnectionError):
            print("连接断开")
            break
        finally:
            sock.close()
        print()

    print("退出")


if __name__ == "__main__":
    main()
