# Record types

Every record declares exactly one `type` in frontmatter. The type selects the skeleton in
[templates/](../templates/) and the filename prefix, and the documentation check rejects a value
outside the declared list.

## The shared header block

Every record opens with a title and a header block. `AGENTS.md` requires a record to state time,
background, cause, action, and effect; that is this block.

```markdown
# <title>

**时间**: <YYYY-MM-DD>
**背景**: <the situation that made this activity necessary>
**起因**: <the specific trigger>
**行动**: <what was done or decided>
**影响**: <what is different afterwards>
```

`**状态**` is retired. Lifecycle state is the directory the record lives in
([lifecycle.md](lifecycle.md)); a free-text status line cannot be read by a machine and drifts from
the location.

A record with no lifecycle state left to express keeps the five lines above and nothing else in the
header block.

## `question` — question sheet

Job: put a bounded set of questions to the developer and record the answers.

Owes: the questions as separate items, each with its own answer area; the developer's answers
recorded verbatim as author statements with the date; and no conclusion the developer did not
state. A question sheet that argues a position is a `discussion`.

The exploration that produced the questions is written as continuous prose without Markdown. Only
the questions use Markdown, each followed by an answer area.

Never fabricate an answer, and never paraphrase the developer's answer into a stronger claim than
they made.

## `discussion` — proposal

Job: argue a feature, requirement, philosophy, architecture, or vision question to a position.

Owes: the position, the alternatives considered, the trade actually taken, and the condition that
would reverse the decision. A proposal is never archived while it is live.

## `design` — code design draft

Job: specify the code design before implementation.

Owes: the module boundaries, what each side owns, the data that crosses the boundary, the failure
behavior, and what the design deliberately does not do. A design that only enumerates the current
code is a `report`.

## `plan` — analysis and change plan

Job: analyse the codebase and state the exact change, its evidence, and its acceptance.

Owes: the current behavior as observed in the code, the exact change, the acceptance criteria, and
the compatibility consequence. A plan states a modification to the existing system that reaches a
goal — no more and no less.

A plan never implements a design that conflicts with the existing architecture. A conflicting change
waits until the developer authorises the removal of the old architecture.

## `report` — change and test result report

Job: report what changed and what the tests showed.

Owes: the change as shipped, the exact commands run, the observed result including failures, and
what remains open. A claim about an operation is admissible only if the operation was run; if it
could not be run, name who can verify it instead of asserting the behavior.

## Adding a type

A new type requires all three, together:

1. a distinct skeleton in `templates/`;
2. an unambiguous position in the records tree or a declared owner;
3. a focused check that maps records to it.

Two of the three is not enough. A new value in frontmatter with no skeleton is not a type.
