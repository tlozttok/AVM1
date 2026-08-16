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
- [docs/Agents](docs/Agents) — archived plans and design decisions
- [docs/record](docs/record) — design notes (historical archive)

## Quick Start

```bash
pip install openai python-dotenv pytest
echo "OPENAI_API_KEY=sk-..." > .env

python -m pytest                          # full test suite (MockLMU, no real API calls)
python main.py                            # read image config from debug.json and run
python main.py <image.json> --frames      # print monitor frame summary after run (stderr)
python main.py <image.json> --transcript api.txt  # write full API calls to a file
```

Debug settings live in `debug.json` (`image` / `frames` / `transcript`) so the command line does not need to change (VSCode runs `python main.py` fixed). Example images: `images/demo.json`, `images/calc_driver.json`; application suite: `images/apps/` (see [docs/apps](docs/apps) for how to start them); AI mini-game: `images/games/echo_mine/`. See the "System Images" section of [docs/runtime.md](docs/runtime.md) for the composition and key points.

## Current Status

Implemented:

- 9 instructions (memory read/write/create, ctrl metadata editing, conversation creation, delivery, close, event listening);
- Memory tree and persistence (write-back to file);
- Scheduling state machine: active/dormant/ready, queue scheduling, core interrupt (instruction budget), call-chain budget inheritance;
- Simple delivery (`send_instruction`: content delivered verbatim + wake + "delivered" confirmation, no ICC, no envelope);
- `close_conversation` kernel-level close (end of an individual: PLM state release);
- System image loading (explicit nodes, para, device plugins, init, persist_to, instruction_budget) and the `main.py` / `debug.json` debugging entry point;
- Python executor (`plm.simple` / `plm.python`) and sandbox (messages executed directly as code/expression, no envelope parsing);
- Info devices (conversations / scheduler / memory / monitor / listeners);
- Event model v1 (Core publishes before request events; Memory/device/Core publish after fact events; phase=before/after/all; device `authorize_listener` / `after_event`; Python handlers, owner checks, notify delivery, `$MEM.listeners`);
- Monitor (frames / trail / diff / find / full-text transcript, including reasoning and event summaries);
- Reasoning content return (kept in history, always returned, recorded by monitor, not exposed internally);
- Single-conversation checkpoint (`save_conversation` / `restore_conversation`, input `/save` debug command);
- AVM application suite (courier_office / vault_guard / plm_ledger: images + custom device + out-of-image custom PLM server);
- AVM mini-game echo_mine (GM conversation + ghost child conversation + dice/map devices + plm.python mechanics + HP event guard);
- 298 tests passing (MockLMU, no real API calls).

Under design (not implemented):

- Device-ification: service calls, registries, etc. become device access (`xx.xx.post` / `xx.xx.response` direction; see docs/record/9.md);
- Kernel/user mode separation;
- Link devices;
- Event conditions (the C in ECA), event persistence/replay, event bus, timers, and other remaining event capabilities — later carriers for "conversation-decided dormancy + expected-behavior listeners";
- Name-to-cid conversion tool;
- File passwords and access callbacks;
- Refined error-handling mechanism (json_error / unknown_tool response pairing);
- Non-blocking alternatives to blocking devices (event mechanism).
