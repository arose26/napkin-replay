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

## 3. Resumability saved the sweep; my death-detector was itself broken

The 40-run sweep ran as 4 shards writing one JSON per finished run, resumable by file existence.
That design survived repeated silent process deaths across two days — each relaunch skipped
finished work and resumed mid-shard — and it is the only reason the sweep ever finished.

The last run (`ratio8` seed 3) took three attempts and ~12 hours of wall clock for what is a
4-hour job. Its shard died once at 05:22 and again at 16:55, both times with no traceback. The
second death coincides to the minute with a kernel global-OOM event (the logged victim was an
unrelated 17GB-virtual java process; the kernel logs only its chosen victim, so python's death
is circumstantial, not proven — and I had *raised* memory pressure myself an hour earlier by
starting three more trainers alongside a browser-automation session).

Two real lessons, one of them embarrassing:

**A run that checkpoints only at completion can lose everything.** The second death threw away
72% of a 4-hour run (360k of 500k steps) because the JSON is written at the end. Per-run
granularity was fine for a 10-minute run and far too coarse for a 4-hour one; checkpoint
granularity should track run length, not the convenience of the file layout.

**Detection code needs its own test.** I wrote a watchdog to catch exactly this — wait on the
pid, then report "exited WITH result" vs "exited WITHOUT result" — and it never produced a
verdict: `pgrep -f` also matched the shell wrapper, so it waited on the wrong pids and fell
through silently. What actually told me the sweep had finished was a dumb loop counting result
files. The monitoring I trusted was less reliable than the monitoring I considered too crude to
mention.

**Takeaway:** resumability solves recovery, not detection — and an untested watchdog is not
detection either. Prefer the crudest check that observes the *artifact* (does the file exist
yet?) over a clever one that observes the *process*.
