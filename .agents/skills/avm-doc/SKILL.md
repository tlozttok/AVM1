---
name: avm-doc
description: 'Use when writing, classifying, archiving, rejecting, deleting, or auditing AVM1 development records — question sheets, discussion proposals, code design drafts, analysis-and-change plans, and change or test result reports. Owns the record substrate that every AVM1 document shares: frontmatter, lifecycle-by-location, the write-time supersession check, citation coordinates, the authority model, and calibration duties.'
---

# AVM1 development records

Development records are the accumulating half of AVM1 documentation. They are written once per
activity, never rewritten into a current-state narrative, and answered by query rather than by
reading in order. The curated current-state half is governed by `avm-wiki`.

Records exist so that a reader who was not present can still decide. That is the only standard this
skill applies.

## Table of Contents

- [The one rule that matters most](#the-one-rule-that-matters-most)
- [Scope](#scope)
- [Record types](#record-types)
- [Lifecycle is directory location](#lifecycle-is-directory-location)
- [The write-time supersession check](#the-write-time-supersession-check)
- [Archiving, rejecting, deleting](#archiving-rejecting-deleting)
- [Citations](#citations)
- [What records are not](#what-records-are-not)
- [Calibration](#calibration)
- [Validation](#validation)
- [Detailed references](#detailed-references)

## The one rule that matters most

**Disposition is decided by future decision value, never by count, age, or length.**

A record stays active while its rationale, alternatives, negative guarantees, ownership boundary,
reintroduction condition, or an independent still-valid rejected option can still guide a future
change. It is archived when the shipped decision is complete and its body is unlikely to guide
future work. Its length is irrelevant in both directions.


## Scope

Use this skill for every Markdown file under the records root `docs/private/Agents/`.

Two exemptions:

- **Pure communication contexts are exempt from the HEAD-resolvability rule.** When an agent
  context's task writes no product code and no externally facing document — it only uses documents
  to talk to the developer — its documents may carry session vantage and change narration. See
  [references/citations.md](references/citations.md).
- **Archived records are frozen.** Never edit, move, reformat, translate, or re-verify an archived
  record's body. Archive with a date line only.

The delete area is not readable by an agent. Never open, grep, cite, or summarize anything under
`docs/private/Agents/.deleted/`.

## Record types

Assign exactly one type per record from the declared list. The type selects the skeleton in
[templates/](templates/) and the filename prefix. Declare it in frontmatter as `type:`; the
documentation check rejects a value outside this list.

| Type | Prefix | Job |
| --- | --- | --- |
| `question` | `question-` | Ask the developer a bounded set of questions and record the answers |
| `discussion` | `discussion-` | Argue a feature, requirement, philosophy, architecture, or vision question to a position |
| `design` | `design-` | Specify the code design before implementation |
| `plan` | `plan-` | Analyse the codebase and state the exact change, its evidence, and its acceptance |
| `report` | `report-` | Report what changed and what the tests showed |

A record that genuinely spans two types is two records, or the dominant job wins. Do not invent a
type: adding one requires a distinct skeleton, an unambiguous position, and a focused check, all
three together. See [references/types.md](references/types.md).

Every record carries YAML frontmatter:

```yaml
---
type: plan
description: "What this record decides, and when a future reader should open it."
---
```

`description` is a retrieval interface, not a summary. It states what the record covers and when to
open it, in one or two sentences with searchable terms. It never restates the title, never claims
importance, and never reports status.

Draft types — `question`, `discussion`, and `design` before approval — also carry:

```yaml
owner: <who is responsible for this record reaching a conclusion>
created: <YYYY-MM-DD>
expires: <YYYY-MM-DD | never>
promotes-to: <path | none>
```

An expired record moves to the delete area automatically. `promotes-to` names the record or page
that will absorb the durable conclusion; `none` is valid and means the conclusion stays here.

## Lifecycle is directory location

The lifecycle state of a record **is** the directory it lives in. There is no status field; a
free-text `**状态**` line is not machine-readable and is retired as records are classified.

```text
docs/private/Agents/
  proposed/      a live proposal; never archived
  implemented/   shipped, still earning its place
  rejected/      a losing option kept only as a guardrail
  archived/      complete and unlikely to guide future work; body frozen
  .deleted/      retired; not readable by an agent
```

Rules:

- **A live proposal is never archived.** If it is no longer worth pursuing, reject it with an honest
  reason.
- **A rejection earns its place only while the losing option stays tempting.** When it stops
  preventing a plausible mistake, it is deleted.
- **Archiving adds one line and changes nothing else.** Insert the archival date below the header
  block; never touch the body.
- **A deprecation note belongs to rejected and deleted records, not to archived ones.** It states
  what the record used to claim, why it no longer holds, and where the current owner is.
- Moving a record is atomic with repairing every inbound link in the same change.

Full classes, transitions, and the reason split are in [references/lifecycle.md](references/lifecycle.md).

## The write-time supersession check

Writing a new record does not retire the record it covers. If you do not check, two contradicting
records stay active and nobody can tell which is current.

**Every new record triggers a scoped check of active records on the same decision, mechanism, or
rejected alternative, performed while writing, in the same change.** Do not defer a known match to
a later corpus audit.

Classify each match:

- **Full supersession** — move every unique rationale, alternative, consequence, shipped
  verification evidence, and named coverage gap into the new record, then archive the old one.
- **Partial supersession** — keep the old record and cross-link. Surviving behavior, a current
  contract, a durable format, a compatibility obligation, or an independently still-valid rejected
  alternative each make supersession partial.
- **Rationale that merely moves is not partial supersession.** Being able to copy an explanation
  elsewhere does not make the old record superseded.

**Ask the developer for a reason on every rejection decision.** A record must never say only that an
option was dropped. Record who rejected it and why, and classify by that reason:

- The record became obsolete over time, or a newer record replaced it → delete area.
- The developer explicitly rejected it and gave a reason, or it was rejected after an implementation
  failure → keep it as a guardrail.

## Citations

A citation is a permanent coordinate into history, not a pointer to the latest version:

```text
<path>@<rev>#<start>-<end>!<hash>
```

Four parts are required: file path, revision, character span, and content hash. Because the
coordinate names its own revision, a citation never goes stale — it resolves to exactly what was
written.

Citations live in a per-record sidecar, `<record>.refs.yaml`, never in frontmatter: independent
edits to a page must not turn mergeable prose into an owner-file conflict.

A record may deliberately keep a duplicate that no longer matches its source. **This is a decision,
not a defect**: when a record still needs the old explanation to stand, keep it and leave the
citation on the old coordinate. Record the decision and its reason. When the source changes and the
record has not been re-decided, that is a different state and it is resolved immediately.

Full syntax, sidecar schema, and the lagging/revived/retired states are in
[references/citations.md](references/citations.md).

## What records are not

**No document is the authority for current behavior.** Four surfaces carry four different kinds of
statement, and each is true in its own mode:

| Surface | Carries | Mode |
| --- | --- | --- |
| Code | fact; the current behavior is derived from it | fact |
| Tests | the current expected behavior | expectation |
| Documents | the designed behavior | design |
| Run records | the behavior at that time | history |

So a record contradicting the code is not an error to be reconciled silently — it is a divergence
to be diagnosed and reported. Find the code's creation date, then classify:

- **the code lags the document** — the document states the design, the code has not caught up;
- **the implementation does not match the expectation** — a real conflict, both sides plausible;
- **an unrecorded change** — someone changed the code and left no record.

**Only the third case changes a document.** The first two are reported, not smoothed over.

Date evidence is asymmetric: a date gap between two documents proves nothing about authority, but a
date gap between a document and the code, or between two code revisions, does.

Records are never authority, and neither are archived records — an archived record stays a valid
citation target, but citing it never proves the system behaves that way now.

## Calibration

These rules are judgment rules. Uncalibrated judgment fails in both directions — it deletes durable
records and keeps dead ones — so every judgment rule here carries worked examples at the bar.

<!-- AUTHOR: reverse lists go here. Each entry names a thing that looks like a violation but is
not, and why. The developer writes these; the agent only leaves the slot and reports candidates.
Highest-value slots first: disposition (keep vs archive), and duplicate (re-decide vs keep old). -->

<!-- AUTHOR: calibrated positive and negative examples go here, with the measurement that shows
size is not the test — one archived record, one kept record, one genuinely borderline record. -->

**Borderline cases are reported, never silently decided.** A case is borderline only when two
dispositions both satisfy the rules above and trade different principles, and this skill does not
already resolve the trade. A record with one defensible answer is not borderline.

Every borderline case triggers one of two repairs, and always produces an example:

- the rule is wrong → change the rule, and write the case as an example of that rule;
- the rule does not reach it → write a new rule, and write the case as an example of the new rule.

Report each borderline case with its measurement and its chosen outcome. These cases are few-shot
samples; optimizing the sample set is continuous work, not an archive.

## Validation

Run before treating a classification or a record as done:

```bash
python scripts/check_doc_links.py     # broken and dangling links
python scripts/check_records.py       # frontmatter, type, location, sidecar
```

Report: records kept, archived, rejected, deleted; every supersession found and its class; every
borderline case with its measurement and outcome; and every dead citation repaired or deleted.

A zero-hit result from either check proves nothing until the check has been calibrated against a
known positive — see [references/calibration.md](references/calibration.md).

## Detailed references

- [references/types.md](references/types.md) — the five types, their skeletons, and what each owes.
- [references/lifecycle.md](references/lifecycle.md) — classes, transitions, archive/reject/delete.
- [references/citations.md](references/citations.md) — coordinates, sidecars, lagging and revival.
- [references/calibration.md](references/calibration.md) — examples, reverse lists, borderline flow.
- [templates/](templates/) — one skeleton per record type.
