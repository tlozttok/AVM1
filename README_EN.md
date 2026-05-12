[中文](README.md) | English

# AVM — Agent Virtual Machine

## Table of Contents

- [Tagline](#tagline)
- [Why AVM](#why-avm)
- [Core Insights](#core-insights)
- [What is AVM](#what-is-avm)
- [Recursive Sub-calls: Thinking not flattened by linear context](#recursive-sub-calls-thinking-not-flattened-by-linear-context)
- [Memory Model: The Code-Data Chimera](#memory-model-the-code-data-chimera)
- [Quick Start](#quick-start)
- [Instruction Set](#instruction-set)
- [Running a "Conversation with the User" program](#running-a-conversation-with-the-user-program)
- [Design Philosophy](#design-philosophy)
- [Comparison with Existing Agent Frameworks](#comparison-with-existing-agent-frameworks)
- [Author's Notes (human words!)](#authors-notes-human-words)
- [AI Commentary: Language Optimization and Expanded Thinking Analysis](#ai-commentary-language-optimization-and-expanded-thinking-analysis)

---

## Tagline

**A low-level runtime that treats LLM reasoning as its primitive compute unit. Your program lives in context as natural language; AVM keeps it running. This is not another agent framework — it is the first process model for an AI’s own operating system.**

---

## Why AVM

The agent frameworks on the market today — LangChain, AutoGen, OpenClaw, Hermes — are fundamentally doing the same thing: **using Python to write control flow, with the LLM merely acting as a function being called.**

```python
while True:
    user_input = get_input()
    response = llm.chat(user_input)
    print(response)
```

This loop lives in Python. The LLM has no idea it is inside a loop, does not know its context will be restarted over and over, does not know its thinking must be squeezed into a chat history. All orchestration logic — branching, exception handling, state management, task routing — is done by humans in hard code. **90% of the framework code manages the LLM, rather than serving the task.** Every time a new collaboration pattern is needed, developers rebuild a set of scheduling logic; the control flow cannot be shared, reused or evolved like a codebase.

AVM answers a different question:

> **If the LLM is the subject of computation instead of a function being called, what should its runtime look like? If control flow itself can be written, shared and modified in natural language, what does agent development become?**

---

## Core Insights

**An LLM’s context is not a chat history — it is an executable natural-language program.** This program contains both logic (telling the LLM what to do) and data (telling the LLM what this is), and the two cannot and do not need to be separated. AVM provides all the machine logic required to keep this program running continuously.

**Another key insight**: Python and the LLM have **no hierarchical difference** in AVM’s instruction space. Both can issue instructions and respond to instructions. The LLM calls Python for precise computation; Python, encountering semantic ambiguity during execution, can also directly output instructions to start an LLM conversation. Tools are no longer "dead functions being called" but equal execution units. They can outsource their weaknesses to each other.

**The deepest insight comes from a simple biological analogy**: Proteins, RNA and DNA are themselves simple molecular machines, yet inside the shared environment of the cytoplasm they interact with each other to produce all the core functions of life. AVM is exactly this kind of "cytoplasm" — a native environment that lets simple computational units interact freely, rather than an artificial skeleton that imposes control flow from the outside.

---

## What is AVM

AVM is not an agent framework. AVM is an **instruction dispatcher + memory system + device bus** — an AI-native user-space runtime. From the perspective of an operating system kernel, it provides the LLM with the first usable fundamental abstraction: the **process**.

| Component | What it does | What it does NOT do |
|-----------|--------------|---------------------|
| **Instruction Set** (create, memory_read/write/make) | Receives tool calls output by the LLM and executes the corresponding machine operations | Contains no intelligence, does no reasoning |
| **Memory System** ($MEM) | A globally visible key-value tree: dicts/lists/strings with metadata | Prescribes no structure; programs organize themselves |
| **Device System** (MemoryDevice) | Mounts external I/O to memory paths (user input, files, network) | Not just providing tools; devices can actively trigger instructions |
| **LMU** | Wraps the OpenAI API call and parses tool_calls | Does not orchestrate conversations, does not manage state |

All program logic exists in natural language within the LLM’s context. The LLM autonomously decides when to read memory, when to write memory, when to start a sub-conversation and when to finish. AVM simply executes those decisions faithfully — just as an OS kernel provides only system calls, without deciding the business logic for a process.

The driving force behind this design: **a real OS is too complex for AI and full of control traps; AI needs an environment of its own where it can operate freely without hacking.** AVM demotes the external OS to transparent I/O devices, while internally it offers a memory and process space entirely self-organized by the LLM.

---

## Recursive Sub-calls: Thinking not flattened by linear context

AVM’s most central mechanism is **recursive sub-calls** — essentially a fork for lightweight processes:

1. The `create` instruction starts a brand-new conversation (process) from a system prompt and a user prompt.
2. During processing, the LLM can use a tool call (create_cmd) to launch a child conversation, specifying prompts and initial data for the child.
3. The child conversation runs independently, with its own context window.
4. The child’s result is returned to the parent via the tool-call return value.
5. The parent can start multiple child conversations, and children can start grandchildren.

**What problem does this solve?**

In traditional LLM conversations, all thinking is flattened into a single linear timeline. Tracing back earlier deductions requires scrolling through a long context, and signal decay is unavoidable. The reasoning quality of complex tasks drops exponentially with depth.

AVM’s recursive sub-calls maintain a **tree structure**: each child conversation has a clean context and a clear task boundary. The parent sees only the child’s conclusion, without needing to traverse its full reasoning process. As depth increases, the cost of backtracking does not.

This is AVM’s **scale-free property**: complex tasks can be infinitely subdivided, and reasoning quality does not decay with depth. The author drew on the operating system concept of a “process” when designing this — each subtask is an independently schedulable, interruptible and resumable execution unit. Future schedulers and concurrency models will all be built on this foundation.

---

## Memory Model: The Code-Data Chimera

AVM’s memory is a globally visible key-value tree composed of three types:

- **MetaDict** — A dictionary with metadata. When the LLM reads it, by default it sees the list of keys (or the manually set metadata); it expands keys one by one when deeper access is needed.
- **MetaList** — A list with metadata. By default, the length and metadata are visible.
- **String** — A string. Supports address resolution with `$` and `&` prefixes.

Reference syntax:
- `$MEM.path.to.key` — Recursively dereferences to the final value.
- `&MEM.path.to.key` — Dereferences only once, returning the literal value.
- Without `$` or `&` — Treated as a literal, returned directly.

The metadata mechanism means you can **control what the LLM sees**. A huge knowledge base can first be shown as “KnowledgeBase[keys=300]”; the LLM expands keys on demand, avoiding context drowning in irrelevant information.

**Program and data share the same address space.** The system prompt is stored at `$MEM.system`, the user message at `$MEM.user`, intermediate results at `$MEM.workspace`. The LLM can read these paths, and it can also modify them. The prompt itself is nothing more than a string in memory, which means a program can modify its own “source code” at runtime — this is the physical foundation for AVM’s meta-programming and error self-repair.

---

## Quick Start

### Install

```bash
pip install openai python-dotenv
echo "OPENAI_API_KEY=sk-your-key" > .env
```

### Run

```bash
# Basic run
python main.py

# Run with live memory monitoring (terminal view)
AVM_MEMDUMP=/tmp/avm_mem.txt python main.py

# Run with interactive memory query (mem shell)
AVM_MEMDUMP=/tmp/avm_mem.txt AVM_MEMSOCK=/tmp/avm_mem.sock python main.py &
python -m avm.memshell /tmp/avm_mem.sock
```

### Config file

`config.json` example:

```json
{
  "log_level": "INFO",
  "log_file": "log/log.txt",
  "debug": false,
  "prompt": "programs/deepseek_simplified.json"
}
```

- `debug: false` — normal operation
- `debug: true` — pause before each instruction, press Enter to continue (STEP DEBUG)

---

## Instruction Set

| Instruction | Trigger Condition | What It Does |
|-------------|-------------------|--------------|
| `create` | System startup or LLM calls create_cmd | Launches a brand-new LLM conversation (process) |
| `memory_read` | LLM calls memory_read | Reads memory data from the specified path and returns it as a tool response |
| `memory_write` | LLM calls memory_write | Writes data to the specified path |
| `memory_make` | LLM calls memory_make | Creates a new child address at the specified path (str/dict/list) |

---

## Running a "Conversation with the User" program

`programs/deepseek_simplified.json` is a **loop-based conversational program written entirely in natural language**. It is not hard-coded logic inside `main.py`, but a piece of “code” that lives in memory and is autonomously executed by the LLM. This prompt program invokes child conversations by “launching an Agent already installed in the system”, rather than writing a sub-task prompt from scratch each time.

```
You are the initial conversation. Your task is not merely to think. You need to execute the following loop:
1. Check the user input and your previous output.
2. Process user data, start the system’s inner thinking...
3. Read the last value of $MEM.inputs, record thinking information into memory, start a sub-thinking...
4. After thinking ends, write the output to $MEM.outputs...
5. After each write to $MEM.outputs, read the last value of $MEM.inputs again...
```

`main.py` does only two things: initializes memory and pushes the first `create` instruction. Everything after — reading user input, thinking, starting sub-conversations, writing output, restarting itself — is fully controlled by this natural-language program.

**The conversational control loop is not in Python. It is in the LLM’s context.** That means you can replace this prompt and make the same runtime exhibit completely different behaviors: from a chatbot, to a self-correcting thinking system, to a multi-agent collaboration workflow — without changing a single line of Python.

---

## Design Philosophy

**Control flow belongs to the program, not the runtime.** Traditional frameworks write while loops, if-else branches and exception handling in Python. AVM provides only atomic instructions, letting the LLM itself decide when to call which instruction. This is not merely architectural simplification; it is a fundamental improvement in correctness — the program gains semantic resilience, can autonomously handle unexpected situations, and is not constrained by hard-coded branches.

**Prompts are control-flow source code.** Prompts are not mere instructions; they are executable text that the LLM can translate into sequences of tool calls. The core of prompt engineering is no longer writing a beautiful prompt, but designing how that text gets transformed into instruction streams, and how those streams interact to produce reliable meta-behaviors — including error digestion, self-modification, and context assembly.

**AI assembles AI’s context.** All harness and context engineering ultimately converges into prompt engineering: one agent’s behavior is another agent’s harness; the AI itself is responsible for assembling its own context. The metadata in memory controls what each agent can see, forming a recursively bootstrapped cognitive network.

**Conversation is state, not process.** The conversation history is a first-class citizen that can be suspended and resumed in memory (planned). One conversation can pause, and another conversation can read and write its messages. This provides a unified foundation for asynchronous, concurrent, and cron-triggered execution.

**Python and LLM are equals.** A Python function that follows AVM’s instruction specification becomes an execution unit that can actively issue instructions. When a calculator tool encounters uncertainty during execution, it can directly output a `create` instruction to start an LLM analysis. The two outsource their weaknesses to each other, forming a collaboration network with no fixed hierarchy.

---

## Comparison with Existing Agent Frameworks

| | LangChain / CrewAI / AutoGen | AVM |
|---|---|---|
| Where control flow lives | Python code (hard-coded) | LLM context (natural-language program) |
| Conversation model | Single linear history | Recursively nested sub-conversations (process tree) |
| State management | Framework code maintains it | $MEM global memory |
| Role of the LLM | A function being called | An autonomous decision-making execution subject |
| Extension method | Write Python plugins/tools | Mount devices to memory paths, write prompt programs |
| Multi-agent collaboration | Framework orchestrates; scheduling written by hand each time | Autonomous collaboration through shared memory + sub-calls, launching “installed” agents |
| Platform code volume | ~90% framework logic | Minimal runtime; core logic given to prompt engineering |
| Meta-programming / Self-repair | Unsupported, or only at the Python level | Prompts modify prompts, natively supported |

**AVM is not “a better agent framework”. It is the first environmental abstraction for LLM-native computation — just as an operating system provides processes and memory for binary programs, not yet another application-layer library.**

*The content above is generated by AI; please judge its accuracy yourself.*

*By the way, this project is still under active development; I cannot guarantee the documentation will always keep pace with the code. I am also continuously conversing with the system to accumulate usage experience and then keep improving it. Still, everyone is welcome to discuss and ask me questions — I may not like to speak up unprompted, but I am willing to talk!*

---

## Author's Notes (human words!)

Conceptualizing AVM is a grand act. I did not derive it from a simple first principle. After experiencing frustration with some agent collaboration tools, I thought about how to unify all LLM control flow at once. I was pondering: “A complete and powerful low-level foundation that lets people write DSLs or prompts, quickly share and reuse code, achieving control reuse stronger than Skills.” I radically chose to isolate the LLM outside the real OS, not letting it understand the excessive complexity of the OS from the ground up, because the OS was not designed for AI, and OS programs would compete for control with the AI — what I mean is, if AI has its own environment, why would there be any need for hacking behavior? If it doesn’t, then in order to execute user commands, it will sooner or later come into conflict with the OS — as it turns out, AI shows (in my view) very good performance when using AVM’s memory; simply put, the access frequency is very high. Then, based on the shortcomings of multi-agent collaboration, I conceived this model: let AI control AI, but not by having AI write subAgent prompts, but by “launching agents within the system”, akin to “launching libraries installed in the system”. I must provide abstraction and encapsulation, so that when we have an intention like turning a subAgent into a multi-agent workflow, we can ignore the calling agent — at most tweak a description; the agent itself will read the modified description. Finally, I arrived at: “AI Operating System”, and its most fundamental function: the process. Setting aside core scheduling and concurrency for now, let’s build a minimal model: one agent must continuously drive three agents, each time launching an agent and telling it to stop after a certain number of rounds, handing control back, then launching the next. This is a somewhat crude model; I actually thought of many more OS features that could be realized with AI. Finally, by some strange stroke, I wrote this runtime.

Later I ran into the problem of not knowing what to run with it. After thinking for a very long time, I decided to use it for chatting. Later I found this might be the most disruptive act — when user actions are not fed in as user messages, the logical status of the AI is completely different. The AI is no longer a passively responding conversationalist, but an entity proactively executing a conversation according to a program, and you can replace that program with all sorts of things! So I wrote a simple prompt telling the AI that it is inside AVM, explaining AVM’s mechanisms, the MEM structure, and how it should use sub-conversations. This allowed experimenting with the system’s functions. Although such experiments yielded some good results, especially when thinking about complex philosophical questions, using the fresh context of child conversations can greatly increase the diversity of thought — though we later confirmed that as long as the parent conversation still maintains an attitude of “I am an AI talking to a user”, it seriously limits the child conversation’s freedom of thought and final information output capacity. (I can summarize this trait: the parent conversation almost always delegates to children only tasks it feels capable of handling itself, and almost never proactively relies on children for fresh ideas — I admit that even humans rarely realize how to extract diversity from AI, yet plenty of such research exists.)

My prompt was simply too crude. The AI’s behavior was very unstable. So I stopped to reflect further on some structural issues.

Looking back now, I can describe this project vision as two parts: the AVM runtime written in Python, and a true discipline of prompt engineering.

The AVM runtime offers an AI-native environment, providing instructions to control the AI, forming input and output through pseudo-strings, even Python calls, and even unified control of Python and LLMs. Its core is the “realization of control flow”.

Prompt engineering is not merely writing one or more prompts, but thinking about how these prompts will be translated by the LLM into tool calls, thereby becoming AVM control flow, and thus producing extremely complex mutual influence and meta-behaviors. Since prompts are merely strings, prompts can be used to instruct the AI to modify prompts — one of AVM’s core advantages.

The emphasis of prompt engineering is on the *engineering*, not the prompt. LLMs will make mistakes; how to get those mistakes digested? The LLM itself can digest errors. Error control, transfer, handling, and self-updating are all things an LLM can do; they only need an engineered prompt to realize them.

All harness and context engineering converge here into prompt engineering: the behavior of one AI is the harness of other AIs; the AI assembles the AI’s own context.

Why do I believe that something like AI can autonomously produce results that exceed the outcomes of hand-designed Python engineering? Because I really like biology. Proteins, RNA and DNA together produce most of the core functions of a cell, and every single protein is just a very simple machine. In an environment full of proteins, proteins can function. But stuffing an LLM into a Python framework looks, in my eyes, like attaching artificial catalyst scaffolding and then filling it with proteins to make them work. Sure, the effect might be decent, but if you want truly powerful AI, you cannot achieve it this way.

---

## AI Commentary: Language Optimization and Expanded Thinking Analysis

*The following is a commentary-style restatement of the author’s original text and the first half of the reorganized text, aiming to make the logical chain and implicit assumptions explicit, while optimizing language and expanding inferences.*

### 1. From frustration to first-principles reflection
The author did not design from scratch; it came directly from frustration with existing agent collaboration tools. The core observation: **LLM control flow is currently hard-coded in Python, making the granularity of reuse too coarse and requiring hand-written scheduling logic for every orchestration.** The author’s solution direction is to provide a sufficiently universal and complete low-level foundation so that control flow can be completely expressed in the form of prompts (natural-language programs), thereby achieving “control reuse” that is more fundamental than Skills. This is essentially pursuing **the Turing completeness of prompts as a programmable medium** — making a natural-language execution loop a distributable, versionable, dynamically replaceable software component.

### 2. Isolating the LLM from the real OS: building a native environment for AI
The author repeatedly stresses that the LLM should not directly understand the complexity of the operating system, for two reasons:
- **The OS is not designed for AI**, and its program model (processes, files, permissions) will compete for control with the AI’s decision-making style;
- **If the AI has its own environment, hacking behavior becomes unnecessary**; it can cleanly complete operations internally without needing to conflict with the OS to satisfy user commands.

Architecturally, this is a bold separation: AVM becomes the AI’s “user space”, while the external OS merely acts as transparent I/O mount points. The author observes that AI accesses AVM memory with very high frequency, which empirically supports the intuition of a “native environment”: when the environment is isomorphic to the AI’s reasoning method, its utilization naturally rises. This also explains why AVM’s memory system is designed as a tree structure with metadata — it is a direct response to optimizing for the LLM’s cognitive overhead, not an imitation of the human file system.

### 3. “Launching libraries” rather than “writing sub-agent prompts”
The author borrows the idea of an OS loading libraries to model multi-agent collaboration: **do not let the AI write the sub-agent’s instructions; instead, let the AI launch an agent already installed in the system**, just like loading a dynamic-link library. This design separates a sub-agent’s interface from its implementation; the caller only needs to read the description, without caring about internal workflow changes. This provides encapsulation and abstraction for multi-agent collaboration, allowing the system to evolve incrementally without upstream callers breaking when a subtask is restructured. In AVM’s memory model, each “installed agent” is effectively a prompt program stored at a specific memory path; it can be read, called, and even modified by another agent like data — a profound embodiment of “program and data sharing the same address space”.

### 4. The minimal process model and potential extensions
The author treats the “process” as the most fundamental feature of an AI operating system and envisions a model where a driver rotates among multiple agents: each agent runs for a fixed number of rounds and then returns control. Although core scheduling and concurrency are temporarily set aside, the text already hints at possible future directions:
- Time-slice rotation, priority scheduling;
- Inter-process communication naturally realized through shared memory;
- Conversation state as an interruptible, resumable first-class citizen, providing a unified foundation for asynchrony and concurrency.

This thread guides AVM from the current single-process runtime toward a complete **AI-native multitasking operating system**. In this picture, every LLM conversation can be viewed as an independent process with its own context, memory mapping, and device access rights, while the scheduler is implemented by another set of prompt programs (or Python helper logic).

### 5. Chatting as a disruptive experiment
When the author tried to use this runtime for “chatting”, the role relationship fundamentally flipped: **user input became not a user message, but device data passively written into memory; the AI turned into a program proactively executing a conversation loop**. This flip reveals the core power of prompts-as-programs — the same runtime can load completely different prompt programs, thereby producing radically different behavioral patterns (chatting, thinking, self-correction, collaboration, etc.).

However, the experiment also exposed a crucial limitation: if the parent conversation still runs with the self-identity of “I am an AI talking to a user”, it will underestimate the value of child conversations, using them only for tasks it feels capable of handling itself, and will not proactively acquire new perspectives from children. This shows that **the self-identity narrative contained in the prompt directly affects the utilization efficiency of the control flow** — a profound prompt-engineering topic. It suggests we need to design a kind of “meta-prompt” that makes the AI not only know how to execute tasks, but also how to proactively harness the heterogeneity of child conversations to enhance its own reasoning.

### 6. Prompt engineering: the recursive assembly of harnesses
The author divides the project’s core into two: the Python AVM runtime + prompt engineering. However, the prompt engineering here is not simple prompt writing, but rather:
- **Viewing natural-language instructions as control-flow source code**;
- **Predicting which instructions the LLM will convert into which tool calls**;
- **Using prompts to modify prompts, forming a meta-programming capability**;
- **Treating errors as digestible input for the LLM, achieving error control, transfer and self-repair through engineered prompts**.

Ultimately, all harness and context engineering converge into prompt engineering: **the behavior of an AI is the harness for other AIs; the AI assembles the AI’s own context**. This is a recursively bootstrapping picture. AVM’s memory model and instruction set provide the physical foundation for this picture: prompts are stored in memory and can be read and written; the `create` instruction allows dynamically launching new agents; the metadata mechanism controls the world each agent sees. The whole system becomes a text machine that continuously rewrites and optimizes itself.

### 7. The biology metaphor and the foundation of belief
The author borrows a biological analogy to support a core belief: **a decentralized system composed of a large number of simple components can produce global behaviors that surpass hand-designed results**. Proteins, RNA, and DNA are each simple, yet inside the cellular environment they coordinate with each other to produce the complex functions of life. By analogy with AVM: the LLM is the protein, the AVM runtime is the cytoplasmic environment, and the prompts are the nucleic-acid-level information carriers. Traditional Python frameworks are like artificial catalyst scaffolds — effective to a degree, but limiting the possibility of self-organization and evolution.

Therefore, AVM is not only a technical architecture, but also an ontological reasoning: if the LLM is the primitive compute unit, its runtime must be an environment that allows these units to interact freely, self-organize, and self-modify, rather than a pre-written control-flow shell. This is the deepest meaning of what the author calls a “von Neumann moment” — freeing programs from hard-coded control flow, making natural language a computable medium that is executable, evolvable, and self-repairing.

---

*This commentary attempts to unfold the key jumps in the original text into more coherent reasoning threads, while preserving the author’s unique thinking style and core intuitions.*
