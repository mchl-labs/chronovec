# ChronoVec for VIBE

[VIBE](https://github.com/vector-index-bench/vibe) is the maintained successor
to ann-benchmarks, which now states it is no longer maintained and points
there. VIBE runs 31 algorithms over 19 modern embedding datasets, 384 to 5120
dimensions, 187K to 20M vectors.

## Why the static number comes from here rather than from us

VIBE measures build and static query. That is the axis this project is weakest
on, and the axis every reader checks first. Measuring it with a harness written
by the project being measured invites exactly the objection it deserves --
during development this project's own comparison was found rigged three
separate times, each in a different direction and each caught only by checking
against something external.

So the static recall/QPS curve is produced by VIBE, and this repository does
not publish its own version of it.

## What VIBE does not measure

Everything ChronoVec exists for: updates, deletions, sustained churn, readers
running while a writer works, snapshots, durability, and space held under
mutation. VIBE indexes are built once and queried. Those axes are measured by
[`streambench`](../../streambench), which is in this repository precisely
because no standard harness covers them.

The two are complementary and neither replaces the other.

## Installing

```bash
git clone https://github.com/vector-index-bench/vibe
cp -r integrations/vibe/chronovec vibe/vibe/algorithms/chronovec
```

Then follow VIBE's instructions to build the container and run:

```bash
python run.py --dataset <name> --algorithm chronovec
```

## What the grid sweeps

`page_capacity` is the build parameter and `nprobe` the query one. The capacity
is swept over 128/256/512/1024 rather than fixed: how it sits against the
data's own cluster structure moves recall per probe substantially, and a single
value would report that guess rather than the index.

Consolidation runs inside `fit`, not before the first query, so the build timer
covers it. A bulk load otherwise leaves every page its splits produced, which
would understate build time and overstate the index's size.

## Testing without a container run

`tests/test_vibe_integration.py` exercises the adapter against a reproduction
of VIBE's `BaseANN`, including the relative import layout, so a typo is caught
in a second rather than in a container build.
