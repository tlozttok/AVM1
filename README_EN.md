# AVM — A Runtime with LLM Reasoning as Its Primitive Compute Unit

AVM is a runtime: LLM inference is the smallest schedulable compute unit, programs are stored in memory as natural-language prompts or Python code, and AVM is responsible for scheduling and executing them.

AVM is not an agent framework. It does not prescribe orchestration patterns, does not ship business prompts, and does not interact with the operating system, hardware, or network. It provides an instruction set, a memory system, a scheduler, a device interface, and a format for composing all of these into a "system image". AVM is self-contained: it has its own memory model and its own computation/scheduling model.

Terminology is defined in [CONTEXT.md](CONTEXT.md).

[中文](README.md)

## Document Index

- [CONTEXT.md](CONTEXT.md) — glossary (conversation, child conversation, sub conversation, link, name, end of interaction vs. end of individual, etc.)
- [docs/runtime.md](docs/runtime.md) — detailed runtime reference (computation model, instruction set, memory model, inter-conversation communication, Python programs, system images, debugging and observability)
- [docs/programming-guide.md](docs/programming-guide.md) — designing AVM programs
- [docs/philosophy.md](docs/philosophy.md) — design philosophy
- [docs/system-image.md](docs/system-image.md) — system image format specification
- [docs/apps](docs/apps) — AVM application suite (practical test images: courier_office / vault_guard / plm_ledger)
- [docs/games](docs/games) — AVM games (echo_mine)
- [docs/examples/python_as_conversation.md](docs/examples/python_as_conversation.md) — example: how a Python program is represented losslessly as "conversation + tool calls"

## Quick Start

```bash
pip install openai python-dotenv pytest
echo "OPENAI_API_KEY=sk-..." > .env

python -m pytest                          # full test suite (MockLMU, no real API calls)
python main.py                            # read image config from debug.json and run
python main.py <image.json> --frames      # print monitor frame summary after run (stderr)
python main.py <image.json> --transcript api.txt  # write full API calls to a file
```

Debug settings live in `debug.json` (`image` / `frames` / `transcript` / `checkpoint_in`) so the command line does not need to change (VSCode runs `python main.py` fixed; `--debug-config <file>` selects another config file). Example images: `images/demo.json`, `images/calc_driver.json`; application suite: `images/apps/` (see [docs/apps](docs/apps) for how to start them); AI mini-game: `images/games/echo_mine/`; the image that reads this repository is `images/repo_tester.json` (the conversation reads tracked files through the read-only `$MEM.AVMRepo` device, whose usage is stated by the device itself). See the "System Images" section of [docs/runtime.md](docs/runtime.md) for the composition and key points.

### Debug Panel

The debug panel is a separate project under `web/` (Vite + React): it observes a running Core and intervenes through the panel conversation. That conversation is an **operator conversation** — its executor is held by the panel rather than a model, and submitted instructions enter the Core verbatim; pausing goes through the interrupt vector to that conversation (`POST /api/interrupt`), resuming uses `POST /api/resume`, and while the panel hosts the run the input device takes lines from the panel box instead of the terminal.

```bash
python scripts/run_console.py images/demo.json   # start the backend; / is the panel once built, otherwise a build hint
cd web && npm install && npm run build   # build the panel into web/dist (not committed)
cd web && npm run dev                    # development: Vite proxies /api to AVM_CONSOLE_URL (default http://127.0.0.1:8700)
```

`scripts/run_console.py` is equivalent to `python -m avm.console`, except that it first gives this process a cleaned `NO_PROXY` / `no_proxy`: on machines with a proxy tool, an already-bracketed IPv6 entry such as `[::1]` makes httpx unable to build a client, and the console dies before its first model call. The script only touches its own process environment; the parent shell and `.env` are left alone.

The panel itself: `cd web && npm run test:unit` runs the pure-logic cases, and `cd web && npm run e2e` starts a scripted-MockLMU console and drives five scenarios in a browser (observation, tool picker, three-step interaction, input feed, console-hosted build).

