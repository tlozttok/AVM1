# Lifecycle and location

The lifecycle state of a record **is** the directory it lives in. There is no status field. Moving a
record between these directories is the state change, and it is atomic with repairing every inbound
link in the same change.

```text
docs/private/Agents/
  proposed/      a live proposal; never archived
  implemented/   shipped, still earning its place
  rejected/      a losing option kept only as a guardrail
  archived/      complete and unlikely to guide future work; body frozen
  .deleted/      retired; not readable by an agent
```

The delete area carries a leading dot so default searches skip it. An agent never opens, greps,
cites, or summarizes anything under `.deleted/`.

## Classes

### `proposed` — a live proposal

A proposal that is still worth pursuing stays here and is **never archived**. If it is no longer
worth pursuing, it does not become `archived`; it is rejected with an honest reason.

### `implemented` — shipped, still earning its place

Split by future decision value, never by count:

- **Keep here** while the record's rationale, alternatives, negative guarantees, durable or wire
  semantics, ownership boundary, security rule, or reintroduction condition is likely to guide a
  future change. Length does not matter.
- **Archive** when the shipped decision is complete and its body is unlikely to guide future work —
  a one-off presentation change, a narrow adapter, a minor closed bug, superseded implementation
  detail. Length does not matter here either.

### `rejected` — a guardrail

A rejection earns its place **only while the losing option stays tempting**. The record must explain
why it loses. When the option is obsolete, superseded, no longer plausible, or unlikely to prevent
re-litigation, it is deleted.

### `archived` — frozen

Archiving adds one line and changes nothing else:

```markdown
**归档**: <YYYY-MM-DD>
```

Inserted below the header block. After that the body is read-only history: never edit, move,
reformat, translate, or re-verify it. An archived record stays a valid citation target, but it is a
historical snapshot, not the authority for current behavior.

Never repair a link out of an archived record, and never fix a stale fact inside one.

### `.deleted` — retired, unreadable

Deletion here is never a real deletion. The record moves to `.deleted/` and records its deletion
date and reason:

```markdown
**删除**: <YYYY-MM-DD>
**删除原因**: <author-stated reason | superseded by <record> | natural obsolescence>
```

A **deprecation note belongs to rejected and deleted records, not to archived ones.** It states what
the record used to claim, why it no longer holds, and where the current owner is now.

## Splitting an in-progress record

A record that is still in progress may have individual sections marked obsolete. When the record
itself completes, split it: each section with a different destination moves to a record of its own
in the matching class. A record is not split while it is still in progress.

## The rejection reason decides the destination

Ask the developer for a reason on every rejection decision. A record must never say only that an
option was dropped.

| Reason | Destination |
| --- | --- |
| Became obsolete over time | `.deleted/` |
| Replaced by a newer record | `.deleted/` |
| Developer explicitly rejected it, with a stated reason | `rejected/` |
| Rejected after an implementation failure, retrospectively | `rejected/` |

The distinction is the point: a rejection that still guards against a mistake the project could
repeat is worth keeping, and a rejection that merely records a dead end is not.

## Supersession

Writing a new record does not retire the record it covers. Every new record triggers a scoped check
of active records on the same decision, mechanism, or rejected alternative, performed **while
writing, in the same change**. Do not defer a known match.

- **Full supersession** — move every unique rationale, alternative, consequence, shipped
  verification evidence, and named coverage gap into the new record, then archive the old one.
- **Partial supersession** — keep the old record and cross-link it. Surviving behavior, a current
  contract, a durable format, a compatibility obligation, or an independently still-valid rejected
  alternative each make supersession partial.
- **Rationale that merely moves is not partial supersession.** Being able to copy an explanation
  somewhere else does not make the old record superseded.

## Migration

The records tree currently holds documents flat, with free-text `**状态**` lines. Classifying a
record means:

1. decide its class by future decision value;
2. move it into the matching directory;
3. replace the `**状态**` line with whichever dated line the class requires, or nothing;
4. repair every inbound link in the same change.

Classify by reading the record, not by its filename date. A date gap between two documents proves
nothing; a date gap between a record and the code does.
