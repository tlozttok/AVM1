# AVM — A Runtime with LLM Reasoning as Its Primitive Compute Unit

AVM is a runtime: LLM inference is the smallest schedulable compute unit, programs are stored in memory as natural-language prompts or Python code, and AVM is responsible for scheduling and executing them.

AVM is not an agent framework. It does not prescribe orchestration patterns, does not ship business prompts, and does not interact with the operating system, hardware, or network. It provides an instruction set, a memory system, a scheduler, a device interface, and a format for composing all of these into a "system image". AVM is self-contained: it has its own memory model and its own computation/scheduling model.

Terminology is defined in [CONTEXT.md](CONTEXT.md); this document only restates terms where the meaning is unambiguous.

[中文](README.md)

## Table of Contents

- [Computation Model](#computation-model)
- [Instruction Set](#instruction-set)
- [Memory Model](#memory-model)
- [Inter-Conversation Communication](#inter-conversation-communication)
- [Python Programs](#python-programs)
- [System Images and Quick Start](#system-images-and-quick-start)
- [Debugging and Observability](#debugging-and-observability)
- [Decisions and Rationale](#decisions-and-rationale)
- [Designing AVM Programs](#designing-avm-programs)
- [Current Status](#current-status)
- [Document Index](#document-index)

## Computation Model

### Conversation Loop

A conversation is the smallest schedulable execution unit. It consists of a Settingup program (prompt text, or a `type="python"` code node) and one startup user message. It runs in a loop:

1. An executor performs one inference (an LLM call, or a call into a Python executor);
2. The output is text, tool calls, or both;
3. When there are tool calls, Core executes the instructions and appends each tool response to the conversation's input batch;
4. When a result without tool calls is produced, the current interaction ends and the conversation goes dormant.

Tool responses, delivered user messages, and user content added by the conversation itself are all merged into the message list before the next inference.

### Scheduling States

A conversation is in one of three scheduling states:

- **Active**: currently performing inference. There is exactly one active conversation at a time.
- **Dormant**: voluntarily suspended, waiting for an event or to be called again.
- **Ready**: queued and waiting to be scheduled.

"Finished" is a conceptual state: it records conversations closed by the kernel-level `close_conversation` instruction (end of an individual). Core's scheduling state machine only has active, dormant, and ready.

### Queue Scheduling and Core Interrupt

Scheduling follows a queue model:

- Woken conversations go to the **tail** of the ready queue (e.g., `call_service`, `send_instruction`, and `return_result` waking the caller);
- `create_sub` is the only exception: the sub conversation is inserted at the front and executes immediately;
- When a sub conversation finishes, its parent is activated directly (same call chain).

To keep a single conversation from holding the execution turn indefinitely, each activation has an instruction budget (`instruction_budget`, default 50, configurable in the image `meta`). Core checks the budget **after tool responses have been appended to the batch**: when the budget is reached, the current conversation is preempted, placed at the tail of the ready queue, and yields. The budget is inherited along the call chain: a sub conversation shares its parent's budget (`is_sub` with a matching parent); a child conversation created by `create_cmd` has an independent budget. When the call chain breaks (an unrelated conversation is picked), the count resets.

### End of an Interaction vs. End of an Individual

"Conversation end" has two meanings that must be kept apart:

- **End of an interaction**: the conversation produced a result without tool calls and the current round is done. The conversation itself still exists and can be woken again (service call, event callback, etc.). It becomes dormant.
- **End of an individual**: the conversation is closed by the kernel-level `close_conversation` instruction (callable by itself or another conversation; the closed conversation's consent is not required). After closure it cannot be scheduled or called again and is recorded as finished; its service is unregistered, its PLM state is released, and its pending calls are cleaned up.

## Instruction Set

Core executes instructions; it does not make business decisions. The full instruction set:

| Instruction | Effect | Key parameters |
|---|---|---|
| `memory_read` | Read memory; returns the LLM-facing representation | `ref` |
| `memory_write` | Write memory | `ref`, `content` |
| `memory_make` | Create a new memory address | `ref`, `key`, `mem_type`(str/dict/list) |
| `edit_metadata` | Edit a node's ctrl metadata: set writes / get reads (omit key to return the whole ctrl) / del removes | `ref`, `type`(set/get/del), `key`?, `value`? |
| `create_cmd` | Create a child conversation (`system_ref` must point to a settingup or python program node; create only, return cid; the child sleeps awaiting instructions; this conversation stays active) | `system_ref`, `para_ref` |
| `create_sub` | Create a sub conversation (executes immediately; on completion, auto-writes the result and wakes the parent) | `system_ref`, `user_ref`, `para_ref` |
| `register_service` | Register this conversation as a service | `name`, `what`, `needs`, `returns` |
| `call_service` | Call a registered service and wait for its return | `service_name`, `input`, `return_mode`(tool/message) |
| `transfer_service` | Transfer control to a service; this conversation sleeps and does not wait | `service_name`, `input` |
| `return_result` | Explicitly return a conclusion to the requester by ICC id and make it the next active conversation | `content`, `icc_id` |
| `send_instruction` | Deliver an instruction to a conversation by cid (multicast supported) | `cid`(int/list), `content`, `wait`, `return_mode`, `format` |
| `close_conversation` | Kernel-level close: close the conversation with the given cid (self or another; consent not required); after closure it cannot be scheduled or called again | `cid` |

## Memory Model

Memory is a global shared key-value tree, addressed with paths like `$MEM.a.b.c`. Three data kinds:

- **str**: leaf node; the content is returned directly.
- **MetaDict / MetaList**: composite types with dual metadata.
- **device**: pseudo-nodes mounted at paths (see below).

### Dual Metadata

MetaDict / MetaList carry two pieces of metadata:

- Metadata one (`meta`, a string): a summary/description for fast scanning by the LLM. By default the LLM sees only the summary and the key list; child nodes are returned only after explicit indexing.
- Metadata two (`ctrl`, a `dict[str,str]`): structured control information for Core/kernel, e.g. `type="para"` (model parameters), `type="settingup"` (LLM prompt program), `type="python"` (Python program).

### Strings and Links

**A str is always a str**: string values are never interpreted as references. The deprecated `$` recursive dereference and `&` single-level dereference do not exist in the current code.

**A link is a device** (design decided, not implemented): it mounts a real path under an alias, e.g. `$rooms` → `$MEM.game.map.rooms`. A conversation reads the target through the alias but never sees the real path, so it cannot use the link to reach sibling nodes — this is how capability boundaries are implemented. Reads follow the target; circular links raise an error.

### Devices

A device is a subclass of `avm.memory_device.MemoryDevice` that interacts with conversations through its interface:

- `to_llm_string()`: the representation an LLM sees when reading the device;
- `resolve_path(path)`: sub-path access (optional; raises by default);
- `set_value(value)`: writes (optional).

Core mounts built-in read-only info devices: `$MEM.conversations` (conversation list and per-conversation detail), `$MEM.scheduler` (active/ready/dormant and budget), `$MEM.icc` (pending ICC records), `$MEM.memory` (memory tree summary), `$MEM.monitor` (frame summary). Images can mount their own devices, e.g. `$inputs`, `$outputs` (see [System Images](#system-images-and-quick-start)).

## Inter-Conversation Communication

### ICC Records

Every inter-conversation call (`call_service`, `send_instruction`) is recorded by Core:

```
icc_id → { caller_cid, caller_call_id, mode, group_key? }
```

The `icc_id` is the caller's tool-call `call_id` for that invocation. `return_result` must carry the `icc_id`; Core routes the conclusion back to the caller using the record. A missing record is an error (printed to stderr).

### Message Format (v2)

Delivered messages are JSON-format strings (the default; the `format` parameter reserves room for future formats, currently only `json`):

```json
{ "from": "sender identity", "to": "target identity", "icc_id": "...", "content": "instruction content" }
```

`from` and `to` are filled in by Core from conversation identities; they are not written by the AI itself. `content` is not required to be JSON.

### Return Modes

Two return modes (`return_mode`) are supported:

- **Tool return (default)**: the `return_result` conclusion is delivered as a tool response attached to the caller's original `call_id`; the caller is woken and placed at the tail of the ready queue.
- **Message return**: the conclusion is delivered to the caller as a user message (`from`/`to`/`icc_id`/`content`); the caller's tool call first receives a confirmation response so that the "one tool call, one return" protocol constraint is satisfied.

For multicast (`send_instruction` with a cid list), each target gets an independent ICC record (`call_id#i`); in tool-return mode Core waits for all targets, then merges the results by cid and name into one JSON-array tool response that Python programs can decode with `json.loads`.

**The parent is not a default receiver**: a creation relationship does not constitute a return channel. Whether a child returns is decided by the child itself via `return_result`; an AI's thinking is not by default revealed to other AIs.

### Names and cids

- A name comes from the `name` field of the Settingup node (node data, not metadata). Names are not unique; collisions are disambiguated by file address.
- A named conversation necessarily has a Settingup file; a sub conversation (created with a direct prompt) has no name, and its identity is the parent's identity plus its own cid.
- `send_instruction` addresses conversations by cid only. Name-to-cid conversion is a kernel responsibility (tool not implemented).

## Python Programs

Python programs are peers of LLM conversations: both are first-class execution units in AVM, communicating through OpenAI-compatible message formats (input: a messages list; output: `content` + `tool_calls`).

A memory node with `ctrl.type="python"` declares a Python program; `content` holds the code. The `model` parameter in `para` selects the PLM (Python Language Model) type:

| model | Behavior |
|---|---|
| `plm.simple` | Calculator: evaluates a single-line expression. Restricted namespace (safe builtins + `math`), stateless. |
| `plm.python` | Turing-complete: Jupyter-notebook semantics. System messages and user messages are both executed into the same persistent namespace (kept across rounds); all `print`s execute and their output is concatenated in order into one result. |

The `model` of a Python conversation must be a name in the registry (`PLM_REGISTRY`); anything else raises an error — no silent fallback to an LLM.

### Determinism

Python program output is deterministic: the executor compares the previous input with the current input and processes only newly added messages; only the return of the last message is kept (the same input sequence always yields the same output). When a message carries an `icc_id`, the concrete PLM automatically constructs a `return_result` (tool-call construction is the concrete PLM's behavior; the base class does not guarantee it). Tool-call ids are a deterministic hash of the message content.

### Sandbox

PLM code runs in a restricted environment: a safe-builtins whitelist plus AST checks (no imports, no dunder attribute access, no dangerous call names such as `open`/`eval`/`exec`). This prevents Python programs from accessing the host system (files, network, processes, clock, entropy sources); behavior inside AVM is not restricted. True strong isolation (subprocess + OS-level) is outside the VM's scope.

## System Images and Quick Start

One JSON file plus a few device Python files is a standard AVM system image. The full spec is in [docs/system-image.md](docs/system-image.md). Key points:

- **Explicit nodes**: every memory node is `{ "kind": "str"|"dict"|"list"|"device", "meta"?, "ctrl"?, "value" }` — no implicit conversion (even str must be wrapped explicitly). `kind="device"` is a readability marker allowed at any level; the actual mount happens in the `devices` section.
- **para**: a dict node with `ctrl.type="para"` holds model-call parameters (`model`, `temperature`, `use_tool`, `extra_body`, `reasoning_effort`, etc.), all stored as strings; numeric values are converted back on load.
- **devices**: an array of `{ path, file, class, args }`. `file` is a Python module path relative to the image file, `class` is the device class defined there, and it **must subclass** `MemoryDevice` or loading fails. This is how "one JSON plus a few device Python files" composes into a complete system image.
- **init**: startup conversation config (`name` defaults to `"init"`; `system`/`system_ref`, `user`/`user_ref`). Only conversation 0 is started; no other conversations are preset — services register themselves at runtime.
- **persist_to**: memory write-back path after the run (also on exception).
- **meta**: `version` must be 1; `instruction_budget` configures the core-interrupt budget.

```bash
pip install openai python-dotenv pytest
echo "OPENAI_API_KEY=sk-..." > .env

python -m pytest                          # full test suite (MockLMU, no real API calls)
python main.py                            # read image config from debug.json and run
python main.py <image.json> --frames      # print monitor frame summary after run (stderr)
python main.py <image.json> --transcript api.txt  # write full API calls to a file
```

Debug settings live in `debug.json` (`image` / `frames` / `transcript`) so the command line does not need to change (VSCode runs `python main.py` fixed). Example images: `images/demo.json`, `images/calc_driver.json`.

## Debugging and Observability

### Monitor

One frame is recorded after every conversation advance; frame 0 is the baseline. A frame contains:

- `lmu`: last execution result (`result`), reasoning content (`reasoning`, truncated like `result`), tool-call summary (`tool_calls`), elapsed time, error;
- `sched`: active cid, ready queue, dormant queue, next cid;
- `conversations`: per-conversation state, message count, and batch contents.

Queries include `trail()` (per-variable trajectory across frames), `diff()` (changes between two frames), and `find()` (frames matching a predicate); `--transcript` writes the full text of every API call to a file.

### Reasoning Content

The `reasoning_content` (chain of thought) from LLM responses is stored in full in the conversation history and sent back with assistant messages in subsequent requests (empty string when absent) — this is a hard requirement of vendor protocols and the chosen "return all reasoning content" policy. The monitor records it (same truncation policy as ordinary content); `$MEM` info devices do not expose it (leak control works by prohibiting internal access, not by not recording).

### Errors

Tool-call argument JSON parse failures, unknown tool names, and similar errors go to stderr for the operator and are not fed back to the LLM for self-correction. The overall error-handling mechanism is still under design (see [Current Status](#current-status)).

## Decisions and Rationale

This project has no precedent, so each design choice below records the reasoning process rather than conclusions first.

### Control Flow Resides in Prompts

Existing frameworks write control flow in Python and treat the LLM as a called function; changing a collaboration pattern means rewriting the program. Prompts are text stored in memory, so a conversation can read, write, and modify them — the physical basis for metaprogramming and error repair. AVM's choice is not "prompts vs. code" but *which layer holds the control flow*.

### Conversations as Processes

A linear context grows with every round; revisiting early reasoning means scanning long history, and the signal decays. A process tree isolates contexts: every conversation has a clean task boundary, a parent sees only the child's conclusion, and increasing depth does not increase the cost of backtracking. This maps the OS process concept onto the LLM runtime.

### Different Wake Behavior for Child vs. Sub Conversations

The two collaboration kinds differ in coupling. A sub conversation is an extension of its parent's control, so it wakes the parent on completion and writes its result back automatically. A child conversation is an independent unit responsible to the overall memory purpose of AVM; the parent goes dormant after creation and waits for an instruction, and the child's completion is not a wake event for the parent. A parent resumes via the message delivered by `return_result` (direct wake) or a registered event.

### A str Is Always a str; Links Are Devices

If a string value could be interpreted as a reference, what is stored and what is read back would differ, and the indirection would be uncontrollable. A link is an explicit device: alias mounting, sub-path delegation, cycle detection — and it carries the capability boundary, because a conversation only receives the alias, never the real path, and cannot reach sibling nodes through it.

### Returning via an Explicit Tool

The LLM API requires exactly one return per tool call. If conclusion delivery depended on a conversation "naturally finishing in the right shape", the outcome would be unreliable: an LLM always produces content but does not always produce a tool call. `return_result` makes returning an explicit action decided by the child conversation. After the call, the child receives a confirmation and ends the next round with a tool-free result, entering dormancy; the caller becomes the next activated conversation.

### End of an Interaction Means Dormant

The end of an interaction is not the end of the individual: the conversation persists and can be called again or woken by an event. Making the terminal state dormant is consistent with asynchronous scheduling — a suspended caller is not in the runnable queue and returns to ready only when an event arrives.

### Errors Face the Operator

System errors during debugging need a human to see them; LLM self-correction would mask runtime defects. Tool argument parse failures and unknown tool names go to stderr rather than back to the LLM.

### Explicit Image Format

An image must reproduce system state exactly. Implicit conversion ("JSON auto-becomes memory types") is convenient but uncontrollable: types, metadata, and para recognition all become guesswork. Explicit nodes (`kind`/`meta`/`ctrl`/`value`) give every field a definite meaning at the cost of verbosity — a price worth paying for a test program.

### Dual Metadata in Memory

LLM context is limited; large files cannot be stuffed into it whole. Metadata one gives the LLM a fast scan (description + key list), with child nodes expanded only on deeper access; metadata two gives Core/kernel the structure (`type`, `signature`) that controls how a program is interpreted. This is how the LLM's field of view is controlled.

### Queue Scheduling and the Core Interrupt

Inter-conversation communication dominates the workload, and each communication yields control like an `await`, so queue scheduling (wake to the tail) is fair enough. `create_sub` is front-inserted because a sub conversation is activated immediately by its parent. The core interrupt is checked after tool responses are appended so the preempted state is complete and recoverable; memory reads/writes are not blocking, and blocking devices are expected to notify conversations through an event mechanism (under design).

### Budget Inherited Along the Call Chain

A sub conversation is an extension of its parent (activated on start); if every activation reset the budget, recursive calls could continue indefinitely. Shared chain budgets bound this. A `create_cmd` child is an independent individual with its own budget.

### ICC and the Message Format

Delivering a conclusion to the right conversation requires addressing, so every call has an ICC record, and the `icc_id` is guaranteed by Core (the caller's tool-call `call_id`). The parent is not a default receiver — an AI's thinking is not revealed to other AIs by default, and a creation relationship is not a return channel. `from`/`to` are filled by Core; the AI neither needs nor should forge them.

### Reasoning Content Return

Thinking-mode requests that carry the `tools` parameter must pass `reasoning_content` back unchanged, or the vendor returns 400. AVM stores it in history and always returns it (empty string when absent); the monitor records it for debugging. Reasoning content is to an LLM what debug output is to traditional code — but it is an AI's thinking and is not exposed to other conversations.

## Designing AVM Programs

(This section is a design guide; the user will edit it later.)

### Programs Reside in Memory

A program has two parts:

- **Settingup node**: `content` is the program text (prompt, or code for `type="python"`). The node's `name` field is the conversation's identity.
- **para node**: call parameters such as `model`, `temperature`, `use_tool`.

A conversation starts from a program node referenced by `system_ref` (or a literal for the entry conversation) plus one user message. For `create_cmd` / `create_sub`, `system_ref` must point to a program node with `ctrl.type="settingup"` or `"python"`; pointing it at a str node or an untyped node raises an error. Program text resides in memory, so conversations can read, inspect, and modify nodes — this is the basis of metaprogramming in AVM.

### One Conversation, One Small Task

Each conversation should take on one small, manageable, traceable task. From a probabilistic-model standpoint: a model tends to endorse content it just produced (otherwise it would not have produced it), while another model looking at old content tends to distrust it (otherwise it would not have been brought in). Small task boundaries are the physical means of isolating these two biases; conclusions travel between conversations through explicit messages, not shared context.

### Express Control Flow with Instructions

A conversation does not "execute" anything itself; it does two things: produce text (a result) or call tools (instructions). Core executes the instructions and puts the responses back into the batch. Control flow is a loop protocol written in the prompt: read input → process → call tools → read returns → write output. Behavior boundaries are enforced by the prompt, not by code.

### Choosing a Call Relationship

Choose along three dimensions (control, waiting, return channel):

| Need | Use | Notes |
|---|---|---|
| Create an independent unit, send instructions later | `create_cmd` | Parent stays active; child sleeps awaiting instructions; returns via `return_result` |
| Execute one small thing immediately and get the result | `create_sub` | Parent sleeps; sub conversation auto-writes back and wakes the parent |
| Reuse an already-registered conversation | `call_service` | Caller sleeps waiting for the return |
| Hand over control, no waiting | `transfer_service` | Suited for infinite loops / tail recursion |
| Send a message to a conversation by cid | `send_instruction` | Multicast and two return modes supported |

### The Return Channel Is Not Default

A creation relationship is not a return channel. To get a result, carry an `icc_id` in the instruction message (the tool-call `call_id` is one), and the callee fills it in when returning. Other conversations learn results through signal events (under design).

### Messages Are Strings

A delivered message is just a JSON-format string: `from`/`to`/`icc_id`/`content`. Python programs parse it with `json.loads`; LLMs are constrained by prompts to take the fields from it. Do not invent extra envelope structures — these are the fields.

### Determinism

A Python program produces the same output for the same input sequence; returns of non-final messages are ignored. LLM programs have no such guarantee, so prompts must state branches and termination conditions explicitly to avoid endless loops (the budget interrupt is a backstop, not the normal path).

### Budget Awareness

Tool calls within one activation count toward the call-chain budget (default 50). Do not pile too many tool calls into one loop iteration; hitting the budget preempts the conversation to the tail of the queue.

### Debug with the Monitor

Every advance produces a frame: active/ready/dormant, tool-call summary, result, reasoning, errors. `--frames` prints the summary, `--transcript` writes full text; info devices let conversations query scheduling state too.

## Current Status

Implemented:

- 12 instructions (memory read/write/create, ctrl metadata editing, conversation creation, services, return, delivery, close);
- Memory tree and persistence (write-back to file);
- Scheduling state machine: active/dormant/ready, queue scheduling, core interrupt (instruction budget), call-chain budget inheritance;
- ICC protocol and message v2 (`from`/`to`/`icc_id`/`content`), multicast tool-return merging;
- `return_result` explicit return;
- `close_conversation` kernel-level close (end of an individual: service unregistration, PLM state release, pending-call cleanup);
- System image loading (explicit nodes, para, device plugins, init, persist_to, instruction_budget) and the `main.py` / `debug.json` debugging entry point;
- Python executor (`plm.simple` / `plm.python`) and sandbox;
- Info devices (conversations / scheduler / icc / memory / monitor);
- Monitor (frames / trail / diff / find / full-text transcript, including reasoning recording);
- Reasoning content return (kept in history, always returned, recorded by monitor, not exposed internally);
- 194 tests passing (MockLMU, no real API calls).

Under design (not implemented):

- Kernel/user mode separation;
- Link devices;
- Event registration and dispatch (timers, memory changes, custom signals) and event bus (implemented by AI programs);
- Name-to-cid conversion tool;
- File passwords and access callbacks;
- Refined error-handling mechanism (json_error / unknown_tool response pairing);
- Non-blocking alternatives to blocking devices (event mechanism).

## Document Index

- [CONTEXT.md](CONTEXT.md) — glossary (conversation, child conversation, sub conversation, service, link, ICC id, name, end of interaction vs. end of individual, etc.)
- [docs/system-image.md](docs/system-image.md) — system image format specification
- [docs/Agents](docs/Agents) — archived plans and design decisions
- [docs/record/author-notes.md](docs/record/author-notes.md) — author notes (historical archive)
