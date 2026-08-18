"""napkin-replay: DQN ships as a bundle -- leave one out.

Repo 3 of the napkin-gamemaster series. napkin-returns found that on-policy,
the textbook variance machinery bought ~nothing and DATA REUSE bought 6x. DQN
is the off-policy version of that bet, and it ships as a bundle: a replay
buffer, a frozen target network, double-Q, n-step returns. Courses teach the
bundle; almost nobody says which parts are load-bearing on a given problem.

So: MinAtar Breakout (10x10x4, real Atari mechanics at napkin cost), one DQN
implementation, and a leave-one-out over the bundle -- plus a replay-ratio
sweep asking the reuse question directly: how many gradient steps per
environment step before reuse turns toxic?

    full        replay 100k + target net + double-Q + 3-step   (the bundle)
    online      no buffer: train on the latest 32 transitions
    notarget    target net == online net (double-Q collapses to vanilla,
                which selfcheck asserts rather than assumes)
    nodouble    vanilla max target
    1step       1-step TD instead of 3-step
    ratio.25    one gradient batch every 4 env steps
    ratio4      4 gradient batches per env step        } same bundle,
    ratio8      8 gradient batches per env step        } more reuse

Matched env steps, 5 seeds per recipe (napkin-returns' rule: fewer is noise),
IQM + bootstrap CIs, ties reported as ties.

Usage:
    PYTHONPATH=.deps python3.10 napkin_replay.py selfcheck
    PYTHONPATH=.deps python3.10 napkin_replay.py train --recipe full --seed 0
    PYTHONPATH=.deps python3.10 napkin_replay.py sweep
    PYTHONPATH=.deps python3.10 napkin_replay.py plot
    PYTHONPATH=.deps python3.10 napkin_replay.py gif
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from minatar import Environment

OUT = Path(__file__).parent / "out"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

GAME = "breakout"
GAMMA = 0.99
LR = 2.5e-4
BUFFER = 100_000
BATCH = 32
WARMUP = 5_000
EPS_START, EPS_END, EPS_STEPS = 1.0, 0.1, 100_000
SYNC = 1_000            # env steps between target syncs
EP_CAP = 5_000          # safety cap; flushed as truncation (bootstrapped)
TOTAL_STEPS = 500_000
SEEDS = 5

BASE = dict(replay=True, target=True, double=True, nstep=3, ratio=1.0)
RECIPES = {
    "full": dict(BASE),
    "online": dict(BASE, replay=False),
    "notarget": dict(BASE, target=False),
    "nodouble": dict(BASE, double=False),
    "1step": dict(BASE, nstep=1),
    "ratio.25": dict(BASE, ratio=0.25),
    "ratio4": dict(BASE, ratio=4.0),
    "ratio8": dict(BASE, ratio=8.0),
}


# ---------------------------------------------------------------------- model

class QNet(nn.Module):
    def __init__(self, channels, actions):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(16 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, actions))

    def forward(self, x):
        return self.f(x)


def obs(env):
    """MinAtar bool [10,10,C] -> float32 CHW."""
    return env.state().astype(np.float32).transpose(2, 0, 1)


# --------------------------------------------------------------------- replay

class Replay:
    """Ring buffer. With capacity == BATCH it degrades into 'train on the last
    32 transitions' -- the no-replay arm, same code path."""

    def __init__(self, cap, shape, seed):
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        self.s = np.zeros((cap, *shape), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, *shape), np.float32)
        self.done = np.zeros(cap, np.float32)   # 1.0 only for true termination
        self.m = np.zeros(cap, np.float32)      # rewards summed (gamma^m boot)
        self.n, self.i = 0, 0

    def add(self, s, a, r, s2, done, m):
        j = self.i
        self.s[j], self.a[j], self.r[j] = s, a, r
        self.s2[j], self.done[j], self.m[j] = s2, done, m
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, k):
        idx = self.rng.integers(0, self.n, size=k)
        return (self.s[idx], self.a[idx], self.r[idx],
                self.s2[idx], self.done[idx], self.m[idx])


class NStep:
    """Accumulates transitions into n-step tuples, never crossing an episode
    boundary. Emits (s, a, R, s_boot, done, m): R sums m<=n discounted rewards,
    the target adds gamma^m * (1-done) * Q(s_boot)."""

    def __init__(self, n, gamma):
        self.n, self.gamma = n, gamma
        self.q = deque()

    def push(self, s, a, r, s2, done):
        self.q.append((s, a, r))
        out = []
        if len(self.q) == self.n:
            out.append(self._pop(s2, done))
        if done:
            while self.q:
                out.append(self._pop(s2, done))
        return out

    def _pop(self, s_boot, done):
        R, m = 0.0, len(self.q)
        for k, (_, _, r) in enumerate(self.q):
            R += self.gamma ** k * r
        s, a, _ = self.q.popleft()
        # remaining tuples lose their oldest member; recompute lazily is O(n^2)
        # in n, which is 3. Clarity wins.
        return (s, a, np.float32(R), s_boot, float(done), float(m))

    def reset(self):
        self.q.clear()


# ---------------------------------------------------------------------- train

def td_loss(net, target_net, batch, cfg):
    s, a, r, s2, done, m = (torch.as_tensor(x, device=DEV) for x in batch)
    q = net(s).gather(1, a[:, None]).squeeze(1)
    with torch.no_grad():
        if cfg["double"]:
            best = net(s2).argmax(1, keepdim=True)
            q2 = target_net(s2).gather(1, best).squeeze(1)
        else:
            q2 = target_net(s2).max(1).values
        tgt = r + (GAMMA ** m) * (1 - done) * q2
    return ((q - tgt) ** 2).mean()


def epsilon(step):
    frac = min(1.0, step / EPS_STEPS)
    return EPS_START + frac * (EPS_END - EPS_START)


def train(recipe, seed, total_steps=TOTAL_STEPS, quiet=False):
    cfg = RECIPES[recipe]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = Environment(GAME)
    env.seed(seed)
    env.reset()
    ch, acts = env.state_shape()[2], env.num_actions()
    net = QNet(ch, acts).to(DEV)
    target_net = QNet(ch, acts).to(DEV) if cfg["target"] else net
    if cfg["target"]:
        target_net.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    cap = BUFFER if cfg["replay"] else BATCH
    buf = Replay(cap, (ch, 10, 10), seed)
    nstep = NStep(cfg["nstep"], GAMMA)

    s = obs(env)
    ep_ret, ep_len, returns = 0.0, 0, deque(maxlen=20)
    curve, frac_acc = [], 0.0
    for step in range(1, total_steps + 1):
        if rng.random() < epsilon(step):
            a = int(rng.integers(acts))
        else:
            with torch.no_grad():
                a = int(net(torch.as_tensor(s[None], device=DEV)).argmax())
        r, done = env.act(a)
        ep_ret += r; ep_len += 1
        truncated = ep_len >= EP_CAP and not done
        s2 = obs(env)
        for tr in nstep.push(s, a, r, s2, done or truncated):
            st, at, R, sb, d, m = tr
            buf.add(st, at, R, sb, 0.0 if truncated else d, m)
        s = s2
        if done or truncated:
            returns.append(ep_ret)
            ep_ret, ep_len = 0.0, 0
            nstep.reset()
            env.reset()
            s = obs(env)

        if step > WARMUP and buf.n >= BATCH:
            frac_acc += cfg["ratio"]
            while frac_acc >= 1.0:
                frac_acc -= 1.0
                loss = td_loss(net, target_net, buf.sample(BATCH), cfg)
                opt.zero_grad(); loss.backward(); opt.step()
        if cfg["target"] and step % SYNC == 0:
            target_net.load_state_dict(net.state_dict())

        if step % 10_000 == 0:
            avg = float(np.mean(returns)) if returns else 0.0
            curve.append((step, avg))
            if not quiet and step % 100_000 == 0:
                print(f"  {recipe} seed {seed}  step {step:>7}  "
                      f"return {avg:6.2f}", flush=True)
    return curve, net


# ---------------------------------------------------------------------- sweep

def run_sweep(total_steps, seeds, shard=0, nshards=1):
    OUT.joinpath("sweep").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    todo = [(rec, s) for rec in RECIPES for s in range(seeds)]
    todo = todo[shard::nshards]
    for k, (rec, seed) in enumerate(todo):
        f = OUT / "sweep" / f"{rec}_{seed}.json"
        if f.exists():
            continue
        curve, _ = train(rec, seed, total_steps, quiet=True)
        f.write_text(json.dumps(curve))
        print(f"[{k + 1:2}/{len(todo)}] {rec:9} seed {seed}  "
              f"final {curve[-1][1]:6.2f}  elapsed {(time.time() - t0) / 60:5.1f} min",
              flush=True)
    print("sweep done")


def load_sweep():
    runs = {}
    for f in (OUT / "sweep").glob("*.json"):
        rec, seed = f.stem.rsplit("_", 1)
        runs.setdefault(rec, {})[int(seed)] = json.loads(f.read_text())
    return runs


def iqm(x, axis=None):
    x = np.sort(np.asarray(x, np.float64), axis=axis)
    n = x.shape[-1] if axis in (None, -1) else x.shape[axis]
    lo, hi = n // 4, n - n // 4
    sl = [slice(None)] * x.ndim
    sl[-1 if axis in (None, -1) else axis] = slice(lo, hi)
    return x[tuple(sl)].mean(axis=axis)


def bootstrap_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    stats = [iqm(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return np.percentile(stats, [2.5, 97.5])


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_sweep()
    loo = ("full", "online", "notarget", "nodouble", "1step")
    ratios = ("ratio.25", "full", "ratio4", "ratio8")     # full == ratio 1
    colors = {"full": "#228833", "online": "#ee7733", "notarget": "#cc3311",
              "nodouble": "#4477aa", "1step": "#aa3377",
              "ratio.25": "#999999", "ratio4": "#66ccee", "ratio8": "#ccbb44"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    results = {}
    for rec in RECIPES:
        seeds = runs[rec]
        curves = np.array([seeds[s] for s in sorted(seeds)])
        x = curves[0, :, 0]
        tail = curves[:, x >= 0.9 * x[-1], 1].mean(1)
        results[rec] = dict(iqm=float(iqm(tail)),
                            ci=[float(v) for v in bootstrap_ci(tail)],
                            seeds=[float(v) for v in tail])
        for ax, group in ((axes[0], loo), (axes[1], ratios)):
            if rec in group:
                for c in curves:
                    ax.plot(c[:, 0], c[:, 1], color=colors[rec], alpha=0.15, lw=0.6)
                ax.plot(x, iqm(curves[:, :, 1], axis=0), color=colors[rec],
                        lw=2.0, label=rec if rec != "full" or ax is axes[0]
                        else "full (ratio 1)")
    axes[0].set_title("leave one ingredient out")
    axes[1].set_title("replay ratio: gradient batches per env step")
    for ax in axes[:2]:
        ax.set_xlabel("env steps"); ax.set_ylabel("episode return")
        ax.legend(fontsize=8)

    order = list(RECIPES)
    vals = [results[r]["iqm"] for r in order]
    errs = np.array([[results[r]["iqm"] - results[r]["ci"][0],
                      results[r]["ci"][1] - results[r]["iqm"]] for r in order]).T
    axes[2].bar(range(len(order)), vals, yerr=errs, capsize=3,
                color=[colors[r] for r in order])
    axes[2].set_xticks(range(len(order)), order, rotation=30, ha="right")
    axes[2].set_title("final IQM return, 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=150)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT / 'results.png'} and results.json")


# ------------------------------------------------------------------------ gif

PALETTE = np.array([[60, 60, 70], [200, 60, 40], [240, 200, 60], [80, 160, 220],
                    [120, 200, 120], [180, 120, 200], [220, 220, 220]], np.uint8)


def render_rgb(env, scale=32):
    st = env.state()
    img = np.zeros((10, 10, 3), np.uint8) + 15
    for c in range(st.shape[2]):
        img[st[:, :, c]] = PALETTE[c % len(PALETTE)]
    return np.kron(img, np.ones((scale, scale, 1), np.uint8))


def make_gif():
    from PIL import Image
    print(f"training '{GAME}' full recipe for the gif ({TOTAL_STEPS} steps)...")
    _, net = train("full", seed=0, quiet=True)
    env = Environment(GAME)
    env.reset()
    frames, total = [], 0.0
    for t in range(1500):
        with torch.no_grad():
            a = int(net(torch.as_tensor(obs(env)[None], device=DEV)).argmax())
        r, done = env.act(a)
        total += r
        if t % 2 == 0:
            frames.append(Image.fromarray(render_rgb(env)).convert("P"))
        if done:
            break
    frames[0].save(OUT / "breakout.gif", save_all=True, append_images=frames[1:],
                   duration=60, loop=0)
    print(f"wrote {OUT / 'breakout.gif'}  return {total:.0f}, {len(frames)} frames")


# ------------------------------------------------------------------ selfcheck

def selfcheck():
    env = Environment(GAME)
    env.reset()
    ch, acts = env.state_shape()[2], env.num_actions()

    # 1. env sanity: shapes, and random play scores at least once in 10k steps.
    assert obs(env).shape == (ch, 10, 10) and obs(env).dtype == np.float32
    rng = np.random.default_rng(0)
    tot = 0.0
    for _ in range(10_000):
        r, done = env.act(int(rng.integers(acts)))
        tot += r
        if done:
            env.reset()
    assert tot > 0, "random play never scored"
    print(f"env sane; random play scores {tot:.0f} in 10k steps")

    # 2. replay sampling is uniform: 1000 ids, 30k draws of 32, all counts
    #    within 6 sigma of expectation.
    buf = Replay(1000, (1, 1, 1), seed=1)
    for i in range(1000):
        buf.add(np.full((1, 1, 1), i, np.float32), i, 0, np.zeros((1, 1, 1)), 0, 1)
    counts = np.zeros(1000)
    for _ in range(30_000 // 32):
        _, a, *_ = buf.sample(32)
        np.add.at(counts, a, 1)
    exp = counts.sum() / 1000
    sigma = np.sqrt(counts.sum() * (1 / 1000) * (1 - 1 / 1000))
    assert np.abs(counts - exp).max() < 6 * sigma, np.abs(counts - exp).max()
    print(f"replay uniform: max deviation {np.abs(counts - exp).max():.0f} "
          f"< 6 sigma ({6 * sigma:.0f})")

    # 3. double-Q target == vanilla max target when online == target weights
    #    (so 'notarget' silently degrades double to vanilla, as documented).
    torch.manual_seed(0)
    net = QNet(ch, acts).to(DEV)
    s = torch.rand(64, ch, 10, 10, device=DEV)
    with torch.no_grad():
        vanilla = net(s).max(1).values
        double = net(s).gather(1, net(s).argmax(1, keepdim=True)).squeeze(1)
    assert torch.equal(vanilla, double)
    print("double-Q == vanilla when online and target nets are tied")

    # 4. n-step accumulator: hand-checked values, correct m, no episode crossing.
    acc = NStep(3, 0.9)
    outs = []
    for t, (r, d) in enumerate([(1, False), (2, False), (3, False),
                                (4, False), (5, True)]):
        outs += acc.push(f"s{t}", t, r, f"s{t + 1}", d)
    assert len(outs) == 5
    R0 = outs[0][2]
    assert abs(R0 - (1 + 0.9 * 2 + 0.81 * 3)) < 1e-6 and outs[0][5] == 3
    assert outs[0][3] == "s3" and outs[0][4] == 0.0
    # flushed tail at episode end: m shrinks, done=1, boot state is terminal's
    assert outs[-1][2] == np.float32(5) and outs[-1][5] == 1 and outs[-1][4] == 1.0
    assert all(o[3] == "s5" for o in outs[2:]), "tail must not cross the episode"
    assert not acc.q, "accumulator must be empty after a done"
    print("n-step tuples: values, m, and episode boundaries all correct")

    # 5. gamma^m bootstrap: target for a 2-reward flush uses gamma^2.
    class Const(nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], acts, device=x.device)
    batch = (np.zeros((1, ch, 10, 10), np.float32), np.array([0]),
             np.array([1.0], np.float32), np.zeros((1, ch, 10, 10), np.float32),
             np.array([0.0], np.float32), np.array([2.0], np.float32))
    with torch.no_grad():
        s_, a_, r_, s2_, d_, m_ = (torch.as_tensor(x, device=DEV) for x in batch)
        tgt = r_ + (GAMMA ** m_) * (1 - d_) * Const()(s2_).max(1).values
    assert abs(float(tgt) - (1 + GAMMA ** 2)) < 1e-6
    print("bootstrap discounts by gamma^m for partial flushes")

    # 6. one fixed batch is overfittable (optimizer + loss are wired right).
    torch.manual_seed(1)
    net = QNet(ch, acts).to(DEV)
    tnet = QNet(ch, acts).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    fb = (np.random.default_rng(2).random((32, ch, 10, 10)).astype(np.float32),
          np.arange(32) % acts, np.ones(32, np.float32),
          np.zeros((32, ch, 10, 10), np.float32), np.ones(32, np.float32),
          np.ones(32, np.float32))
    for _ in range(500):
        loss = td_loss(net, tnet, fb, RECIPES["full"])
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 1e-2, float(loss)
    print(f"one-batch Bellman overfit: loss {float(loss):.1e}")

    # 7. epsilon schedule endpoints.
    assert epsilon(0) == 1.0 and abs(epsilon(EPS_STEPS) - 0.1) < 1e-9
    assert epsilon(10 * EPS_STEPS) == epsilon(EPS_STEPS)
    print("epsilon schedule endpoints correct")

    print("\nall selfchecks passed")


# ------------------------------------------------------------------------ cli

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck")
    t = sub.add_parser("train")
    t.add_argument("--recipe", choices=RECIPES, default="full")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=TOTAL_STEPS)
    s = sub.add_parser("sweep")
    s.add_argument("--steps", type=int, default=TOTAL_STEPS)
    s.add_argument("--seeds", type=int, default=SEEDS)
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--nshards", type=int, default=1)
    sub.add_parser("plot")
    sub.add_parser("gif")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if args.cmd == "selfcheck":
        selfcheck()
    elif args.cmd == "train":
        t0 = time.time()
        curve, _ = train(args.recipe, args.seed, args.steps)
        print(f"final return {curve[-1][1]:.2f}  ({time.time() - t0:.0f}s)")
    elif args.cmd == "sweep":
        torch.set_num_threads(2)
        run_sweep(args.steps, args.seeds, args.shard, args.nshards)
    elif args.cmd == "plot":
        make_plots()
    elif args.cmd == "gif":
        make_gif()
