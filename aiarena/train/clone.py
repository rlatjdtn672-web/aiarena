"""베끼기 — 손코딩 봇의 판단을 그대로 흉내 내게 만든다.

맨땅에서 PPO 를 돌리면 판이 어려울수록 한참 0% 에 머문다.
우연히라도 이겨봐야 "이게 좋은 행동이었다"는 신호가 생기는데,
한 번도 못 이기면 배울 거리 자체가 없기 때문이다.

그래서 먼저 베낀다. 손코딩 봇을 따라 하게 만들어 놓고, 거기서부터 PPO 로 넘어간다.
베낀 망은 손코딩을 넘지 못한다 — 넘는 건 그다음 PPO 의 몫이다.

    python -m aiarena.train.clone --teacher kite --steps 3000
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from ..core.spec import SCENARIOS
from ..core.world import World, RESULT_RUNNING
from ..core.obs import build_obs, alive_mask
from ..bots.baselines import BASELINES
from .ppo import Policy, evaluate


def collect(scenario, teacher, foe_names, episodes, seed=4242):
    """선생이 두는 수를 모은다. 상대는 돌아가며 바꾼다 —
    한 상대만 보면 그 상대에서만 통하는 수를 베끼게 된다."""
    X, Y = [], []
    for e in range(episodes):
        foe = BASELINES[foe_names[e % len(foe_names)]]
        w = World(scenario)
        w.reset(seed + e * 7919)
        while True:
            o0, m0 = build_obs(w, 0), alive_mask(w, 0)
            o1, m1 = build_obs(w, 1), alive_mask(w, 1)
            a0 = teacher(o0, m0)
            for i in range(len(o0)):
                if m0[i]:
                    X.append(o0[i])
                    Y.append(int(a0[i]))
            w.apply_actions(0, a0)
            w.apply_actions(1, foe(o1, m1))
            if w.advance() != RESULT_RUNNING:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)


def clone(policy, X, Y, steps=3000, batch=512, lr=1e-3, log_every=500):
    """정책망만 가르친다. 값머리는 손대지 않는다."""
    params = list(policy.body.parameters()) + list(policy.pi.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    xt = torch.as_tensor(X)
    yt = torch.as_tensor(Y)
    n = len(xt)
    for s in range(1, steps + 1):
        mb = torch.randint(0, n, (batch,))
        logits, _ = policy(xt[mb])
        loss = nn.functional.cross_entropy(logits, yt[mb])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % log_every == 0 or s == 1:
            with torch.no_grad():
                acc = (policy(xt[:4096])[0].argmax(-1) == yt[:4096]).float().mean().item()
            print(f"    베끼기 {s:5d}/{steps}  틀린정도 {loss.item():.4f}  따라함 {acc*100:5.1f}%")
    return policy


def warm_value(policy, X, Y, steps=600, lr=1e-3):
    """★ 값머리 예열. 이걸 건너뛰면 PPO 첫 업데이트에서 정책이 망가진다.

    베끼기는 정책만 가르치므로 값머리('앞으로 딸 점수 예상')는 백지로 남는다.
    백지 값머리로 PPO 를 시작하면 되돌림 값이 엉터리로 나오고,
    그 엉터리를 따라 정책이 폭주한다. 정책을 얼려두고 값머리만 먼저 맞춘다.
    """
    for p in policy.body.parameters():
        p.requires_grad_(False)
    for p in policy.pi.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.v.parameters(), lr=lr)
    xt = torch.as_tensor(X)
    target = torch.zeros(len(xt))      # 아직 실제 되돌림이 없으니 0 에서 출발시킨다
    for s in range(steps):
        mb = torch.randint(0, len(xt), (512,))
        _, v = policy(xt[mb])
        loss = ((v - target[mb]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in policy.body.parameters():
        p.requires_grad_(True)
    for p in policy.pi.parameters():
        p.requires_grad_(True)
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="raider_vs_biter")
    ap.add_argument("--teacher", default="kite", choices=list(BASELINES))
    ap.add_argument("--foes", nargs="*", default=["rush", "kite", "spread"])
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", default="runs/clone")
    args = ap.parse_args()

    sc = SCENARIOS[args.scenario]
    os.makedirs(args.out, exist_ok=True)
    print(f"선생 {args.teacher} · 상대 {args.foes} · {args.episodes}판에서 수를 모읍니다")

    X, Y = collect(sc, BASELINES[args.teacher], args.foes, args.episodes)
    print(f"  모은 판단 {len(X):,}개")

    policy = Policy()
    clone(policy, X, Y, steps=args.steps)
    warm_value(policy, X, Y)

    print("\n  베낀 망 실력:")
    for f in args.foes:
        w, l, t = evaluate(policy, sc, f)
        print(f"    vs {f:<7} {w:>2}:{l:<2} (시간초과 {t}) = {w/(w+l+t)*100:5.1f}%")
    print("\n  선생(손코딩) 실력:")
    from ..core.match import run_match
    from ..core.world import RESULT_P0_WIN, RESULT_P1_WIN
    for f in args.foes:
        w = l = t = 0
        for e in range(24):
            r = run_match(sc, BASELINES[args.teacher], BASELINES[f], seed=99000 + e * 7919)
            if r.result == RESULT_P0_WIN:
                w += 1
            elif r.result == RESULT_P1_WIN:
                l += 1
            else:
                t += 1
        print(f"    vs {f:<7} {w:>2}:{l:<2} (시간초과 {t}) = {w/(w+l+t)*100:5.1f}%")

    path = os.path.join(args.out, "cloned.pt")
    torch.save({"model": policy.state_dict(), "teacher": args.teacher}, path)
    print(f"\n★ 저장: {path}")
    print(f"   이어서 학습: python -m aiarena.train.ppo --foe mixed --init {path}")


if __name__ == "__main__":
    main()
