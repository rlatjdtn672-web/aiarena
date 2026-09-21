"""PPO 학습. torch 만 있으면 된다.

돌리는 법:
    python -m aiarena.train.ppo --foe spread --updates 200

한 팀의 유닛 전원이 망 하나를 같이 쓴다. 유닛마다 자기 관측을 넣고 자기 행동을 받는다.
그래서 유닛 수가 달라져도 같은 망을 쓸 수 있다.
"""
import argparse
import os
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn

from ..core.spec import SCENARIOS, OBS_DIM, N_ACTIONS, ACT_NAMES
from ..core.env import ArenaEnv
from ..core.reward import RewardConfig, PRESETS
from ..core.world import RESULT_P0_WIN, RESULT_P1_WIN, RESULT_TIMEOUT
from ..bots.baselines import BASELINES


class Policy(nn.Module):
    """관측을 넣으면 행동 확률과 '앞으로 딸 점수 예상'을 같이 낸다."""

    def __init__(self, obs_dim=OBS_DIM, n_act=N_ACTIONS, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_act)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_np, greedy=False):
        x = torch.as_tensor(obs_np, dtype=torch.float32)
        logits, value = self(x)
        if greedy:
            a = logits.argmax(-1)
            return a.numpy(), None, value.numpy()
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a.numpy(), dist.log_prob(a).numpy(), value.numpy()


def make_bot(policy: Policy, greedy: bool = True):
    """학습한 망을 봇 계약에 맞춰 감싼다. 이대로 리그에 낼 수 있다."""
    def bot(obs, mask):
        a, _, _ = policy.act(obs, greedy=greedy)
        return [int(x) if mask[i] else 0 for i, x in enumerate(a)]
    return bot


def evaluate(policy, scenario, foe_name, episodes=24, seed=99000):
    """foe_name 이 mixed 면 손코딩 3종 전부를 상대로 재서 평균을 낸다."""
    if foe_name == "mixed":
        tw = tl = tt = 0
        for k in ("rush", "kite", "spread"):
            w, l, t = _eval_one(policy, scenario, k, episodes, seed)
            tw += w; tl += l; tt += t
        return tw, tl, tt
    return _eval_one(policy, scenario, foe_name, episodes, seed)


def _eval_one(policy, scenario, foe_name, episodes=24, seed=99000):
    """실력 재기. 학습 중 로그가 아니라 이 숫자로 판단한다 —
    학습 로그의 승률은 일부러 흔들어 뽑은 값이라 실제 실력보다 낮게 나온다."""
    from ..core.match import run_match
    bot = make_bot(policy, greedy=True)
    w = l = t = 0
    for e in range(episodes):
        r = run_match(scenario, bot, BASELINES[foe_name], seed=seed + e * 7919)
        if r.result == RESULT_P0_WIN:
            w += 1
        elif r.result == RESULT_P1_WIN:
            l += 1
        else:
            t += 1
    return w, l, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="raider_vs_biter")
    ap.add_argument("--foe", default="spread",
                    choices=list(BASELINES) + ["mixed"],
                    help="mixed = 판마다 상대를 바꾼다 (한 상대에만 통하는 답을 외우지 못하게)")
    ap.add_argument("--reward", default="기본", help=f"점수표: {list(PRESETS)}")
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--rollout", type=int, default=2048, help="한 번 배울 때 모으는 판단 수")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--eval-every", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sc = SCENARIOS[args.scenario]
    rc = PRESETS[args.reward] if args.reward in PRESETS else RewardConfig()

    if args.foe == "mixed":
        # ★ 한 상대로만 배우면 '그 상대를 이기는 법'이 아니라
        #   '그 상대의 움직임'을 외워버린다. 판마다 상대를 갈아준다.
        _pool = [BASELINES[k] for k in ("rush", "kite", "spread")]
        _rng = np.random.default_rng(args.seed)

        def foe(obs, mask):
            return foe.cur(obs, mask)
        foe.cur = _pool[0]

        def _reroll():
            foe.cur = _pool[int(_rng.integers(len(_pool)))]
        foe.reroll = _reroll
    else:
        foe = BASELINES[args.foe]
        foe.reroll = lambda: None

    env = ArenaEnv(sc, foe, reward=rc)
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    os.makedirs(args.out, exist_ok=True)

    print(f"판 {sc.key} · 상대 {args.foe} · 점수표 {args.reward}")
    print(f"점수: {rc.as_dict()}")
    print(f"유닛 {env.n_units}기 · 관측 {OBS_DIM} · 행동 {N_ACTIONS}\n")

    ep_seed = 1000
    obs = env.reset(ep_seed)
    best = (-1, None)
    t0 = time.time()
    ep_results: List[int] = []

    for upd in range(1, args.updates + 1):
        # ── 판을 돌며 겪은 것을 모은다 ────────────────────────────
        O, A, LP, V, R, D = [], [], [], [], [], []
        for _ in range(args.rollout):
            a, lp, v = policy.act(obs)
            O.append(obs.copy()); A.append(a); LP.append(lp); V.append(v)
            obs, r, done, info = env.step(a)
            R.append(r); D.append(done)
            if done:
                ep_results.append(info["result"])
                ep_seed += 1
                foe.reroll()          # 다음 판은 다른 상대일 수 있다
                obs = env.reset(ep_seed)

        # ── 얼마나 좋았는지 되짚는다 (GAE) ────────────────────────
        n = args.rollout
        nu = env.n_units
        _, _, last_v = policy.act(obs)
        adv = np.zeros((n, nu), dtype=np.float32)
        gae = np.zeros(nu, dtype=np.float32)
        for t in reversed(range(n)):
            nonterm = 0.0 if D[t] else 1.0
            nxt = last_v if t == n - 1 else V[t + 1]
            delta = R[t] + args.gamma * nxt * nonterm - V[t]
            gae = delta + args.gamma * args.lam * nonterm * gae
            adv[t] = gae
        ret = adv + np.array(V, dtype=np.float32)

        # ── 배운다 ───────────────────────────────────────────────
        bo = torch.as_tensor(np.concatenate(O), dtype=torch.float32)
        ba = torch.as_tensor(np.concatenate(A), dtype=torch.long)
        blp = torch.as_tensor(np.concatenate(LP), dtype=torch.float32)
        badv = torch.as_tensor(adv.reshape(-1), dtype=torch.float32)
        bret = torch.as_tensor(ret.reshape(-1), dtype=torch.float32)
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)

        idx = np.arange(len(bo))
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), args.minibatch):
                mb = idx[s:s + args.minibatch]
                logits, value = policy(bo[mb])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(ba[mb]) - blp[mb])
                s1 = ratio * badv[mb]
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * badv[mb]
                loss = -torch.min(s1, s2).mean() \
                       + 0.5 * ((value - bret[mb]) ** 2).mean() \
                       - args.ent * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()

        # ── 어떻게 되어 가는지 ───────────────────────────────────
        if upd % 5 == 0 or upd == 1:
            recent = ep_results[-40:]
            wr = recent.count(RESULT_P0_WIN) / max(1, len(recent))
            to = recent.count(RESULT_TIMEOUT) / max(1, len(recent))
            with torch.no_grad():
                ent = torch.distributions.Categorical(logits=policy(bo[:512])[0]).entropy().mean().item()
            print(f"  {upd:4d}회  판 {len(ep_results):5d}  롤아웃승률 {wr*100:5.1f}%  "
                  f"시간초과 {to*100:4.1f}%  탐험 {ent:.2f}  {time.time()-t0:5.0f}초")

        if upd % args.eval_every == 0 or upd == args.updates:
            w, l, t = evaluate(policy, sc, args.foe)
            rate = w / (w + l + t)
            star = ""
            if rate > best[0]:
                best = (rate, {k: v.clone() for k, v in policy.state_dict().items()})
                torch.save({"model": policy.state_dict(), "upd": upd, "eval": rate},
                           os.path.join(args.out, "best.pt"))
                star = "  ← 역대 최고, 저장"
            print(f"       └ 붙여보기(고정) {w}:{l} (시간초과 {t}) = {rate*100:.1f}%{star}")

    print(f"\n★ 최고 {best[0]*100:.1f}%  →  {args.out}/best.pt")
    print(f"   손코딩 기준선은 아래 표에서 확인: python tools/crosstable.py")


if __name__ == "__main__":
    main()
