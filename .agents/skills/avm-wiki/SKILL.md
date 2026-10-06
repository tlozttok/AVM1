---
name: avm-wiki
description: 'Use when creating, updating, splitting, merging, or auditing AVM1 wiki pages — the current-state documentation layer of README, CONTEXT.md, and docs/*.md. Owns page granularity, the discussion area, outgoing and incoming links, freshness and revival, the automatic index, and the page-graph acceptance measures.'
---

# AVM1 wiki

The wiki is the current-state half of AVM1 documentation. It is updated in place over time. Its job
is to be current, complete, and concise, and to stay that way while the system changes.

Development records — question sheets, proposals, design drafts, plans, reports — are the other
half. They accumulate and are answered by query. This skill covers pages only; the record substrate
is owned by `avm-doc`.

## Table of Contents

- [The one rule that matters most](#the-one-rule-that-matters-most)
- [Scope](#scope)
- [Page types](#page-types)
- [Page granularity](#page-granularity)
- [The discussion area](#the-discussion-area)
- [Links](#links)
- [Freshness, revival, retirement](#freshness-revival-retirement)
- [The index](#the-index)
- [Page graph and acceptance](#page-graph-and-acceptance)
- [Validation](#validation)
- [Detailed references](#detailed-references)

## The one rule that matters most

**A page states only what is currently true, and it is updated in the same change that makes it
false.**

A page never narrates its own history. What changed, when, and why belongs in the automatic index
and in development records. A reader who opens a page gets the present state, not the route to it.

This is why the wiki is cheap to keep correct and why its body carries no version markers: there is
exactly one current answer, and updating it is the whole maintenance action.

## Scope

Use this skill for the current-state documentation layer: `README.md`, `README_EN.md`, `CONTEXT.md`,
and `docs/*.md` outside `docs/private/`.

The uploaded layer is a **processed and reduced** version of the local corpus, not a second
standard. A local document may be long and complete; its uploaded counterpart is the part a reader
outside this machine needs. Both carry frontmatter; the difference is scope, not format.

Two exemptions:

- **Product documentation is not record documentation.** The writing discipline in `avm-doc`'s
  citations reference applies to records. Product docs state the design as it is, without session
  vantage, and do not need a rationale trail.
- **A page with no inbound links is not automatically wrong.** A page may exist because a concept
  appears repeatedly across the wiki and needs one central statement. That is an inbound-link
  argument, not a defect.

Never write a page by hand into an index. The index is generated.

## Page types

A page declares its type in frontmatter; type selects the page's shape and is recorded by the
automatic index.

| Type | What it holds |
| --- | --- |
| `code-dev` | Developer documentation for a module: what it is, its interface, its invariants |
| `image-dev` | Developer documentation for a system image: nodes, wiring, expected behavior |
| `model` | An abstract design model: a concept the system instantiates |
| `philosophy` | A design principle or philosophical position, with its consequences |
| `example` | A design example, including unimplemented examples kept deliberately |
| `index` | An index page over a group of documents, classified for retrieval |

## Page granularity

A page is the right size when it has:

- its **own owner** — one party responsible for keeping it current;
- its **own change cadence** — it changes when its subject changes, not when a neighbour does;
- its **own inbound links** — other pages point at it for one reason;
- its **own validation** — a check that can tell whether it is still true.

A concept that appears repeatedly across the wiki needs a central page even if nothing links to it
yet. That is the fifth criterion and it is a sufficient reason on its own.

**One fact, one owner.** When two pages state the same proposition, the philosophy page owns it and
`AGENTS.md` cites it rather than restating it. Establishing this across the existing pages is
ongoing work, not a precondition for writing new ones.

### Splitting

Splitting a page is justified **only when its parts show different clustering features** — different
citation targets, different cited-by objects, or different modification-time characteristics. A page
that is merely long is not split; length alone is never the test.

Use `avm-doc`'s citation coordinates for anything the split would duplicate, so the new pages can
each point at the historical revision they came from instead of restating it.

## The discussion area

There is **exactly one** discussion area per page, and it is named 讨论区. Nothing else on the page
is provisional: every other section is polished current state.

The discussion area holds directions not yet decided, working hypotheses, and unverified ideas. It
may be sparse, and a page with a decided subject may have none at all.

Close it out when the page's subject settles. The closing actions, in order:

1. promote durable conclusions into the page body — the body is where a decided statement lives;
2. move reusable rationale into a development record, which is what records are for;
3. keep genuine chronology, if it matters, in a postmortem record — never on the page;
4. delete resolved discussion.

A page does not carry its own change history. This is the same rule as the one rule at the top,
applied to the discussion area.

## Links

**An outgoing link is created the moment its target is encountered.** There is no quota to reach and
no upper bound to respect: if a page mentions a concept that has or should have a page, link it in
the same edit. Deferring links is how a wiki decays into isolated prose.

Incoming links are the wiki's load-bearing structure, and they are maintained by the linking side.
Before moving, renaming, splitting, or retiring a page, repair every inbound link in the same change.

Every page explains the pages it relates to, so **duplication across pages is expected**. It is
acceptable when each duplicate carries a reference at the granularity `avm-doc` defines — a path,
a revision, a character span, and a hash. Duplication without a reference is a claim with no
ancestry, and it is what "one fact, one owner" exists to prevent.

## Freshness, revival, retirement

Three states, and they are decisions rather than defects:

- **Lagging** — a page deliberately keeps an explanation that no longer matches its source, because
  the page still needs the old explanation to stand. Record the decision and leave the reference on
  the old coordinate. A lagging reference is a **different edge**, not a degraded one.
- **Revived** — an abandoned idea becomes meaningful again and develops independently. It becomes a
  new page; it does not resurrect the old one. AVM1 itself is an instance of this pattern.
- **Retired** — the page is no longer worth keeping. The developer's judgment sends it to the delete
  area; it is never silently dropped.

The state that is **not** acceptable is **unconfirmed**: the source changed and the page was never
re-decided. That is resolved immediately, in the same change, either by updating the page or by
recording a deliberate lag.

## The index

The index is **generated, never handwritten**. It is built from page structure and metadata, and it
exists so retrieval does not depend on anyone maintaining a list.

For each page it records what the page itself does not carry:

- the kind of content — code-facing or principle-facing;
- the page's structure, owner, and purpose;
- outgoing and incoming links;
- who wrote it — human or AI — and in what context;
- the note the creator left, and the creation date.

**At minimum, the wiki carries a major-change history index.** Ordinary page edits do not belong in
it; a major change is one a reader would need in order to understand why the current state is the
current state.

`AGENTS.md` cites pages; it does not become a second wiki. Cross-domain prerequisites live there.

## Page graph and acceptance

The wiki is accepted by the state of its page graph, not by page count and not by any threshold
copied from another project.

Maintain the graph — pages as nodes, links as edges, with lagging edges distinguished from current
ones — and report the following whenever asked:

- **maximum hop count** between any two pages, which should stay small enough that a reader reaches
  any page in a few steps;
- **clustering**, which should be moderate: neither one undifferentiated blob nor a scatter of
  isolated pages.

Report the measurements, not a verdict. A number that moves without a reason is a finding.

## Validation

Run before treating a page or an index as done:

```bash
python scripts/check_doc_links.py     # broken and dangling links, skill and wiki alike
python scripts/check_records.py       # record frontmatter, type, and location
```

Report: pages created, split, merged, retired; links added and inbound links repaired; lagging and
revived pages with their decisions; and the current graph measures.

A check that reports nothing has proven nothing until it has been calibrated against a known
positive — `avm-doc`'s calibration reference records two failures from this repository, one in each
direction.

## Detailed references

- [references/pages.md](references/pages.md) — page types, granularity, splitting, frontmatter.
- [references/links.md](references/links.md) — out-links, in-links, duplicate references, freshness.
- [references/index.md](references/index.md) — generated index, major-change history, graph measures.
- [templates/page.md](templates/page.md) — the page skeleton.
