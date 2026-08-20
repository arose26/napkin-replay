# Insights from building napkin-replay

Written in the order I hit them, same convention as the rest of the series.

## 1. The famous stabilizers were ties; the boring one was load-bearing

The registered hypothesis had the target network "hurting clearly but survivably" — the
literature's second-most-celebrated DQN ingredient. Measured: `notarget` is a dead tie with
`full` (11.97 vs 12.02, CIs nested). Same for double-Q. The only catastrophic removal was the
replay buffer itself (−58%), and the only other real ingredient was 3-step returns (−25%).

Scale caveat stated plainly: a 10×10 MinAtar game, small conv net, 50k-transition buffer. The
divergence spirals that target networks exist to kill may need bigger values, longer horizons,
or denser bootstrapping to develop. The honest claim is "not detectable at this scale with
n=5", not "unnecessary".

**Takeaway:** the components a method is famous for and the components it dies without are
different lists, and only a leave-one-out tells you which is which at your scale.

## 2. Sample reuse saturates long before compute does

Replay ratio 0.25 starves (−24%), ratio 1 is enough, and ratios 4 and 8 buy nothing more at
matched env steps — while costing 4× and 8× the gradient FLOPs. The knee registered in the
hypothesis is real but sits at 1, not past 4 as predicted: this buffer/net size extracts what a
transition has to give in roughly one visit.

The same arithmetic read in the other direction bit the series' finale repo on the same day:
there, scaling rollout width silently *cut* updates-per-sample 4× below the reference and
stalled learning entirely (see napkin-gamemaster INSIGHTS). Update density has a wide flat
plateau — and a cliff on each side.

## 3. Sharded sweeps need per-run files AND a death watchdog

The 40-run sweep ran as 4 shards writing one JSON per finished run, resumable by file
existence. That design survived three separate silent process deaths (no traceback, no OOM
record — cause never identified) across two days: each relaunch skipped finished work and
resumed mid-shard. What it did NOT survive gracefully was *noticing* the deaths — a dead shard
looks exactly like a slow shard from the outside. The fix that stuck: a watchdog that waits on
the process and reports "exited WITH result" vs "exited WITHOUT result" the moment it happens.

**Takeaway:** resumability solves recovery, not detection. Sharded long jobs need both.
