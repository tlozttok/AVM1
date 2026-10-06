# Calibration

Every rule in this skill is a judgment rule. Uncalibrated judgment fails in **both** directions: it
deletes durable records and it keeps dead ones. Calibration is what makes a judgment rule executable
by a reader who was not present when it was written.

## Examples set the bar

A judgment rule carries worked examples at the bar, with the measurement that shows size is not the
test. Name real records, state the measurement, and give the reason:

```text
archived: <record> — <measurement>: complete, unlikely to guide future work
kept:     <record> — <measurement>: durability boundary still binds
borderline: <record> — <measurement>: still guides <what>, kept
```

Examples may sit inline with the rule or in a `references/examples.md`, whichever keeps the rule
readable. Both are valid; choose per rule.

## Reverse lists

A reverse list names what **looks like a violation but is not**, and why. Without one, a rule that
only says what to flag will over-match: `**状态**` inside a quoted example, a date in a filename, a
citation to an archived record, a duplicate kept deliberately.

<!-- AUTHOR: The developer writes the reverse lists. The agent's job is to leave the slot and to
report candidates it encountered, not to fill it. Highest-value slots first:
  1. disposition — kept vs archived
  2. duplicate — re-decide vs deliberately keep the old explanation
  3. rejection — guardrail vs delete -->

## Borderline

A case is borderline **only** when two dispositions both satisfy the rules and trade different
principles, and this skill does not already resolve the trade. A case with one defensible answer is
not borderline, and reporting it as such is noise.

Every borderline case triggers exactly one of two repairs:

- **the rule is wrong** → change the rule, and write the case as an example of that rule;
- **the rule does not reach it** → write a new rule, and write the case as an example of the new rule.

Either way the case becomes an example. A borderline case is never filed and forgotten.

## Report format

Report every borderline case with:

- the record and its measurement;
- the two dispositions in tension and the principle each trades;
- the chosen outcome;
- which repair it triggered.

## These are few-shot samples

Borderline cases, examples, and reverse-list entries are **samples for future judgment**, not an
archive. Their correct state is to be continuously filtered, rewritten, and replaced. A sample that
no longer changes a decision is removed; a sample that resolves a recurring case is promoted into
the rule it calibrates.

Do not accumulate them for completeness. A sample set that grows without being optimized has stopped
being calibration and become the problem it was meant to solve.

## Calibrate a probe before trusting it

A pattern-based check proves nothing until it has been calibrated against a known positive and a
near-miss negative. Two failures from this repository, one in each direction:

- **over-match** — a Chinese-language battery designed for residue in English prose flagged ordinary
  Chinese words in Chinese documents, and nearly produced the false conclusion that the corpus was
  full of reasoning leakage;
- **under-match** — a link check that matched only `docs/Agents` missed `../Agents`, and a broken
  link survived a verification pass.

Both came from the same cause: the check had never been calibrated. When a check reports nothing,
say so as a calibration gap, not as a clean result.
