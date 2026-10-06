# Pages

## Frontmatter

Every page carries frontmatter. The type selects the page's shape and is recorded by the automatic
index.

```yaml
---
type: philosophy
description: "What this page states, and when a reader should open it."
---
```

`description` is a retrieval interface. It says what the page holds and when to open it, in one or
two sentences with searchable terms. It never restates the title and never reports status.

## Types

### `code-dev` — developer documentation for a module

What the module is, its interface, the invariants it maintains, and the failure behavior callers
must handle. A code-facing page may state behavior that the code currently does not implement, but
then it says so explicitly rather than describing the code as if it already matched.

### `image-dev` — developer documentation for a system image

The nodes in an image, how they are wired, and the behavior the wiring is expected to produce. This
is documentation of an artifact, not of the runtime that runs it.

### `model` — abstract design model

A concept the system instantiates, stated so it can be reasoned about independently of any
implementation. When the system and the model diverge, the divergence is a finding, not a page edit.

### `philosophy` — design principle

A position and its consequences: what follows from holding it, and what it rules out. **The
philosophy page owns any proposition it states.** When other pages need the same proposition they
cite it; `AGENTS.md` cites it too.

### `example` — design example

A worked example of the system, including examples that are deliberately not implemented. An
unimplemented example is marked as such and stays: it records intent that the current state does not
express.

### `index` — index page over a group of documents

An entry point into a classified group of development records. Index pages are **generated from
metadata and directory position**; they are never maintained by hand. A group of records without an
index page is a retrieval gap, not a stylistic preference.

## Granularity

A page is correctly sized when it has all four of the following, or the fifth on its own:

| Criterion | Test |
| --- | --- |
| Own owner | One party is responsible for keeping it current |
| Own change cadence | It changes when its subject changes, not when a neighbour does |
| Own inbound links | Other pages point at it for one reason |
| Own validation | A check can tell whether it is still true |
| Appears repeatedly | The concept recurs across the wiki and needs one central statement |

Length is not a criterion in either direction. A page of one sentence is a page — everything here is
a conversation, and a conversation can be short.

## Splitting

Split **only** when the parts show different clustering features:

- different citation targets — the parts point at different sources;
- different cited-by objects — different pages come to each part for different reasons;
- different modification-time characteristics — the parts change on unrelated schedules.

A page that is long, or that covers several related ideas, is not split on that basis.

Procedure:

1. name the clustering feature that differs, and state it in the change;
2. create the new page or pages with their own frontmatter and descriptions;
3. give every duplicated statement a citation coordinate under `avm-doc`'s scheme instead of
   restating it without ancestry;
4. repair inbound links so each now points at the page that actually holds its answer;
5. leave the original as the owner of the part that remains with it, or retire it if nothing remains.

## Merging

Merge when two pages share an owner, a change cadence, and a citation target, and separate existence
only forces readers to visit both. State the same evidence as for a split. A merge is not the
inverse of a split with a different justification threshold — it needs its own clustering argument.

## One fact, one owner

When two pages state the same proposition, the proposition is owned by exactly one of them: in
practice the philosophy page for design positions, the module page for interface facts. The other
page cites the owner.

Applying this to the existing pages is ongoing work. It is not a precondition for writing a new
page, and it is never done by deleting the citing page's context — a page may keep the explanation
it needs, with a reference.
