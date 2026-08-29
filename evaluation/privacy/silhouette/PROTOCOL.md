# The attack protocol, binding on every silhouette re-identification measurement

Fixed by the project owner on 2026-08-01; R5 added 2026-08-07 after it inverted a
ranking. Every attack must satisfy all five rules. Any result
that does not is a diagnostic, not a threat-model measurement, and must be labelled as such.

---

## The five rules

### R1: The attacker's training data is CASIA-B, and only CASIA-B

They hold our public code, so they can push CASIA-B through our defence and train on the result.
They hold nothing else. No part of our capture corpus may appear in training, in any form.

### R2: Testing is on our dataset, and only our dataset

The question is whether a person can be re-identified from **our** emitted output. A gallery of
CASIA-B people cannot answer that, because our subjects are not among them. So evaluation runs
entirely on our clips.

### R3: RAW video never appears anywhere. Ever.

If a source video was used to produce a silhouette or a generated character, the attacker may
hold the released silhouette or the released generation, because those are what we release. The
raw video must not be in any training set, any gallery, or any probe set. The attacker never sees
the unprotected footage; that is the thing they are trying to recover.

### R4: A probe never matches against its own source clip

A released artifact may not be scored against itself, or against anything else derived from the
same source recording. Otherwise the task is trivial and the number is meaningless.

---

### R5: The defence is applied where it is deployed, never inside a canvas normalised from the clean person

A gait harness normally crops and rescales each silhouette to a fixed canvas before matching. If
that normalisation is derived from the **clean** person and then the defence is applied inside the
already-normalised canvas, the harness silently undoes the part of the defence that changes the
person's extent, and preserves only the part that changes their outline. That flatters exactly the
smooth, outward-only family of defences, because those are the ones whose whole effect survives.

It is not a small effect. Re-running the arms with the defence applied where it actually is
deployed, and ranking by lift over each arm's own measured null, **inverted the ranking**:
`displace` -- the mode that shipped on the strength of the flattering protocol -- turned out to
remove essentially none of the available lift, while `bbox`, which the old protocol ranked poorly,
took the frozen attacker to its own chance floor. `bbox` is what ships now.

So: normalise from the **emitted** artifact, the one a real attacker would hold. `silhouette_harness.py`
imports the deployed mitigation and its deployed parameters from `tier1/src/edge_runner_pi5/`
rather than reimplementing them, so the arm measured is the arm that ships.

---

## What the attack therefore is

It is **linkage**, not identification against a named list. The attacker holds several of our
released outputs and asks "are these the same person?". They cannot put a name to anyone, because
our subjects are not in CASIA-B, but linking two releases is already a privacy failure, and it is
the strongest thing they can actually do.

So: train a feature extractor on defended CASIA-B, embed our released silhouettes with it, match
ours against ours, exclude same-source (R4). Top-1 is how often the nearest neighbour is
genuinely the same person.

The paper's appearance-channel measurement (Classes 1 and 2, in
[`../appearance/`](../appearance/)) instantiates this: training-free GEI matching and the frozen
published GaitBase checkpoint, evaluated on the capture corpus with condition-matched galleries,
same-source exclusion, and a chance floor measured by label permutation through the identical
pipeline.

---

## Two measurement lessons the protocol enforces

**A single-seed number from a learned attacker is meaningless.** Across ten identical training
runs on identical data, a learned transfer attacker's top-1 spanned 6 to 14 percentage points per
defence, which is larger than the gaps between defences. An earlier three-seed run was quoted to
two decimal places and produced a conclusion that reversed on re-measurement, and was withdrawn.
Always report mean, spread and n, and when the number defends a claim, quote the attacker's best
seed, because the attacker chooses their own training seed.

**A cross-domain attacker's weakness is partly domain gap, not only defence strength.** A network
trained on distant full-body walking footage performs poorly on close-range chest-up footage
regardless of any defence. An in-domain learned adversary was measured to beat the training-free
attacker outright, but obtaining in-domain footage breaches R1, so that attacker sits outside
this threat model by construction, not by evidence of weakness. State that limit whenever these
numbers are quoted: they bound the attacker the protocol defines, not all conceivable risk.
