# Index and page graph

## The index is generated

The index is built from page structure and metadata. It is never handwritten.

This is deliberate: a hand-maintained list of pages becomes a second place where the truth about the
wiki lives, and it is wrong the moment somebody forgets it. A generated index cannot drift.

**The index describes what the pages themselves do not carry.** Per page:

| Field | Why it is not in the page |
| --- | --- |
| Content kind — code-facing or principle-facing | A property of the page as a whole; noise in every section |
| Structure | Contents follow from the headings |
| Owner | Changes with organisation, not with content |
| Purpose | That is what `description` is for; the index aggregates them |
| Outgoing and incoming links | Derivable, and stale the moment it is written by hand |
| Author — human or AI | Not a claim the page should make about itself |
| Creation context | What the writing context was doing; meaningless inside the page |
| Creator's note | A remark to the maintainer, not to the reader |
| Creation date | Retrieval metadata, not content |

The author and context fields matter because AI-written and human-written pages need different
review. A page written by an agent during a refactor and a page written by the developer while
thinking through a design are not the same artifact, even when their prose looks alike.

## Major-change history index

**At minimum, the wiki carries an index of major changes.**

A major change is one a reader would need in order to understand why the current state is the current
state: an architecture change, an ownership change, a decision that reversed an earlier one.

Ordinary page edits do not belong here. The page body does not carry change history either — the
history lives in this index and in development records, which is what keeps the page body in the
present tense.

## Index pages over records

A group of development records is entered through an index page classified for retrieval. The
records are archived by class and looked up in detail; the index exists so that retrieval does not
require knowing the class first.

An index page states its group in a few sentences and does not restate its children. Restating them
creates a second copy of every child, and the copy is the one readers will trust.

## Page graph

Maintain the graph and be able to report it at any time:

- **nodes** — pages;
- **edges** — links, with **lagging edges distinguished from current ones**;
- **defects** — unconfirmed edges, which must be zero before any measure is reported.

### Measures

**Maximum hop count.** The largest number of steps between any two pages. It should stay small
enough that a reader reaches any page in a few steps. A maximum that grows means the wiki is
fragmenting into regions that do not link to each other.

**Clustering.** How tightly pages group. It should be **moderate**: neither one undifferentiated blob
where every page links to every other, nor a scatter of isolated clusters. Too much clustering means
the page boundaries are not doing work; too little means the links are not doing work.

### Reporting

Report the measurements with the change that produced them, not as a verdict. A number that moved
without a corresponding change in the wiki is the interesting finding.

Page count is not a measure of anything. There is no target, and no threshold borrowed from another
project applies — another project's "three to five pages" is that project's answer to a different
question.
