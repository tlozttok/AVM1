# Links

The wiki's value is its link structure. A set of accurate pages with no links is a pile, not a wiki.

## Outgoing links

**Create an outgoing link the moment its target is encountered.** There is no quota and no upper
bound.

If a page mentions a concept that has a page, link it. If it mentions a concept that should have a
page, link it and create the page. Deferring a link "until the section is finished" is how a wiki
decays into isolated prose that only its author can navigate.

This replaces an earlier convention of adding a bounded number of links per page. A bound made the
link set a budget to spend rather than a record of what the page actually refers to.

## Incoming links

Incoming links are the load-bearing structure, and they are maintained by the linking side.

Before moving, renaming, splitting, or retiring a page, repair **every** inbound link in the same
change. A page that has been moved with its inbound links left behind is worse than an unmoved page:
the content is correct and unreachable.

Inbound links are also how a page's necessity is diagnosed. A page with no inbound links is not
automatically wrong — it may satisfy the "appears repeatedly" criterion — but it is worth asking
which of the two it is.

## Duplicate statements and references

Every page that explains a related page duplicates something. That duplication is **expected and
acceptable**, on one condition: each duplicate carries a reference at the granularity `avm-doc`
defines — path, revision, character span, hash.

```text
docs/philosophy.md@a1b2c3d#1200-1580!9f2c1e77
```

The reference is what makes a duplicate maintainable. Without it, a duplicate is a claim with no
ancestry, and no reader can tell whether it matches the source or has drifted.

**A reference updating does not require every referencing page to change.** A page may keep pointing
at the old coordinate. That is the lagging state below, and it is history — not merely git history,
but a still-standing earlier version that may later be caught up or retired.

## The three states

### Lagging — a decision

A page deliberately keeps an explanation that no longer matches its source, because the page still
needs the old explanation to stand. The page and its source may both be old and still useful.

Record the decision with its reason, and leave the reference on the old coordinate. The coordinate
still resolves, which is the point of naming a revision.

A lagging reference is **a different edge**, not a degraded one: it points at a historical revision,
while a current reference points at the present one. Graph measures count them separately.

### Revived — independent development

An abandoned idea becomes meaningful again and develops independently. It becomes a **new page**; it
does not resurrect the old page's life cycle, and it is not a resynchronization of the old one.

AVM1 is itself an instance: a computation model that was set aside and later became worth developing
on its own terms.

### Retired — the developer's judgment

The developer decides a page is no longer worth keeping. It goes to the delete area with its
deletion date. It is never silently dropped and never left to rot in place.

## The unacceptable state

**Unconfirmed** — the source changed and the page was never re-decided. This is not a lag; it is an
unanswered question about what the page should now say.

Resolve it immediately, in the same change that revealed it, by either updating the page or recording
a deliberate lag with its reason. Only unconfirmed edges are defects, and they are resolved before
any graph measure is reported.
