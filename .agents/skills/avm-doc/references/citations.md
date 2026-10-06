# Citations

## A citation is a coordinate, not a pointer

A citation names a permanent position in history:

```text
<path>@<rev>#<start>-<end>!<hash>
```

| Part | Meaning |
| --- | --- |
| `<path>` | repository-relative file path |
| `@<rev>` | the revision the citation was written against |
| `#<start>-<end>` | the character span, counted from the start of the file |
| `!<hash>` | a hash of exactly that span |

All four parts are required. Because the coordinate names its own revision, **a citation never goes
stale** — it resolves to exactly what was written, whatever the file says now. There is no
"outdated citation" state to police.

The span is chosen by the writer, not by heading structure. Any character range is a valid span.

Example:

```text
CONTEXT.md@a1b2c3d#1234-1580!9f2c1e77
```

## Sidecar

Citations live in a sidecar beside the record, never in its frontmatter:

```text
docs/private/Agents/implemented/plan-thing-2026-09-01.md
docs/private/Agents/implemented/plan-thing-2026-09-01.refs.yaml
```

```yaml
provenance:
  author: ai | human
  context: "<one line: what the writing context was doing>"
citations:
  - target: CONTEXT.md
    rev: a1b2c3d
    span: 1234-1580
    hash: 9f2c1e77
    state: current | lagging
    reason: "<required when state is lagging>"
```

Hashes never go in the record's frontmatter. Independent edits to a page would change the same header
line and turn otherwise mergeable prose into an owner-file conflict.

`provenance` records what the index needs and the page itself does not carry: who wrote it, and in
what context. It is written once, at creation.

## Three states

### Current

The record agrees with what its citation points at. Nothing to do.

### Lagging — a deliberate decision

A record may keep a duplicate that no longer matches its source. When the record still needs the old
explanation to stand, the old explanation is not an error: the record and its source may both be old
and still useful.

This is a **decision**, not a defect. Record `state: lagging` with the reason. Leave the citation on
the old coordinate — the coordinate still resolves, which is the point of naming a revision.

The alternative is that the source changed and the record was never re-decided. That is a different
state, **unconfirmed**, and it is resolved immediately: either re-decide the duplicate, or record the
deliberate decision.

### Revived

A rejected option can become meaningful again and develop independently. That is a revival, not a
supersession and not a resynchronization. A revived option becomes a new record of its own; it does
not resurrect the old record's life cycle.

## Graph consequences

A lagging citation is **a different edge**, not a degraded one: it points at a historical revision,
while a current citation points at the present one. Two edges with different targets are two edges.

Graph measures — maximum hop count, clustering coefficient — therefore count current and lagging
edges separately. Only unconfirmed edges are a defect, and they are resolved before measurement.

## Resolvability at HEAD

A record must be readable by someone who has the repository and nothing else. For every reference,
ask: **can a reader at HEAD, holding no session transcript, no review thread, and no uncommitted
draft, resolve this reference and verify this claim?**

Failure modes and their repairs:

| Failure | Repair |
| --- | --- |
| Dead session citation — a letter code, a draft section number, a phase label | If a committed owner exists, cite it by name and path; otherwise delete the citation and restate the fact as a standalone sentence |
| Change narration — "used to", "no longer", "the old design" | State the current behavior; write a fixed regression as a present-tense counterfactual ("without X, Y happens") |
| Review choreography — who said what in which round | Keep the surviving decision and its reason as an ordinary fact; delete the attribution |
| Reviewer-addressed justification | State the invariant that makes it correct, or delete it |
| Hedged planning residue | Promote to an explicit `TODO`, or state the actual limit; delete the hedge |

**Exemption.** When an agent context's task writes no product code and no externally facing document
— it only uses documents to talk to the developer — its documents may carry session vantage and
change narration. The exemption is granted by **what the context produced**, not by directory or
filename.

Probes for these patterns are deliberately not shipped yet. Until a calibrated probe exists, an
unpatterned read of the densest prose in scope is the only admissible method; an uncalibrated scan
fails in both directions and must not be used as evidence.
