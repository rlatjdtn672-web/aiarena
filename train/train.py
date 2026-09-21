#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""비대칭 셀프플레이 PPO 학습기.

train/ppo_micro.py 에서 파생했습니다. 원본은 한 줄도 고치지 않습니다.

원본과 다른 점 세 가지가 이 파일의 전부입니다.
  1) 정책망을 2개 씁니다. P0=벌처망, P1=저글링망. 서로 유닛이 달라서 한 망으로 못 합니다.
     각자 자기 옵티마이저·자기 버퍼·자기 GAE 를 갖습니다.
  2) 로그를 metrics.jsonl 에 한 줄씩 붙입니다 (통짜 덮어쓰기 금지).
  3) 환경이 죽으면 그 환경만 다시 띄웁니다. 멈추라는 신호를 받으면 마지막 세대를 저장합니다.

★ 이 파일에서 제일 위험한 것은 P0 와 P1 의 점수·관측·마스크가 슬롯 인덱스로 섞이는 사고입니다.
   한 글자만 틀려도 학습은 멀쩡히 돌고 로그도 멀쩡한데 아무것도 안 늡니다.
   그래서 배선 검사를 코드 안에 넣었습니다:

       python3.11 train_sp.py --selftest

사용 예:
    python3.11 train_sp.py --ally vulture --allies 1 --enemy zergling --enemies 6 \
        --frame-skip 4 --max-steps 350 --envs 8 --updates 400 --out vult_sp1
"""
import argparse, collections, copy, json, os, signal, struct, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RAW_OBS = int(os.environ.get("SC_OBS", "39"))   # ★ 프로토콜 관측 크기. 기본 39,
#   셔틀 판(bwshuttle, 탑승·스카랍·셔틀거리 3칸 추가)처럼 바이너리가 다르면 SC_OBS 로 맞춘다.
# ★ 행동 수는 바이너리마다 다르다. bwdragoon 은 12개, bwctrl 은 14개(어택땅·홀드 추가).
#   안 맞추면 학습기가 새 행동을 아예 못 쓴다.
N_ACT = int(os.environ.get("SC_ACTS", "12"))
OBS_DIM = RAW_OBS             # 망이 실제로 받는 크기. 기억을 달면 여기만 커진다.

# ── 기억 ────────────────────────────────────────────────────────────
#   ★ 왜 필요한가: 박자(택견)는 화면이 아니라 '지금 몇 번째 박자인가' 로 결정된다.
#     화면만 보는 망은 전진할 차례와 후퇴할 차례를 구별할 정보가 없어서,
#     사람이 짠 택견을 **베끼는 것조차** 못 한다(따라 하기 정확도 55.2% = 제일 흔한 답만 찍음).
#     숫자 몇 개를 더 주면 100% 로 베낀다. 실측 근거는 research_star/기록_사람조건_반응속도와APM.md.
MEM_MODE, MEM_DIM = "none", 0
CLOCK_P = (4, 8, 16, 32)      # 박자 후보를 미리 안 알려주려고 2의 거듭제곱만 쓴다(정답 9는 없다)
MEM_K = 5                     # acts 모드에서 기억하는 최근 행동 수


def set_memory(mode):
    """망이 받는 관측을 키운다. 망·버퍼를 만들기 전에 딱 한 번 부를 것."""
    global MEM_MODE, MEM_DIM, OBS_DIM
    MEM_MODE = mode
    MEM_DIM = {"none": 0, "clock": 2 * len(CLOCK_P), "acts": MEM_K * N_ACT,
               "both": 2 * len(CLOCK_P) + MEM_K * N_ACT}[mode]
    OBS_DIM = RAW_OBS + MEM_DIM
    return OBS_DIM


def mem_row(step, hist, n):
    """(n, MEM_DIM). hist[i] = 유닛 i 의 최근 행동 (새것이 앞)."""
    if MEM_DIM == 0:
        return np.zeros((n, 0), np.float32)
    parts = []
    if MEM_MODE in ("clock", "both"):
        f = []
        for q in CLOCK_P:
            f += [np.sin(2 * np.pi * step / q), np.cos(2 * np.pi * step / q)]
        parts.append(np.tile(np.array(f, np.float32), (n, 1)))
    if MEM_MODE in ("acts", "both"):
        o = np.zeros((n, MEM_K * N_ACT), np.float32)
        for i in range(n):
            h = hist[i] if hist is not None and i < len(hist) else ()
            for k in range(min(MEM_K, len(h))):
                o[i, k * N_ACT + int(h[k])] = 1.0
        parts.append(o)
    return np.concatenate(parts, 1)
ACT_NAME = ["정지", "이동1", "이동2", "이동3", "이동4", "이동5", "이동6", "이동7", "이동8",
            "최근접공격", "최저HP공격", "후퇴", "어택땅", "홀드",
            "태우기", "내리기"][:N_ACT]
# ★ 판마다 행동 수가 다르다(셔틀 판은 16개). 이름표가 모자라면 세대 저장 때
#   ACT_NAME[k] 에서 그냥 죽는다. 모자란 자리는 번호로 채운다.
ACT_NAME += [f"행동{i}" for i in range(len(ACT_NAME), N_ACT)]

# 경로는 전부 이 저장소 기준이다. 환경변수로 바꿔 끼울 수 있다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPQ  = os.environ.get("SC_MPQ",  os.path.join(ROOT, "data", "mpq"))
MAP  = os.environ.get("SC_MAP",  os.path.join(ROOT, "maps", "Weave_v1.scx"))
BIN  = os.environ.get("SC_BIN",  os.path.join(ROOT, "engine", "bwmicro_sp"))
EXTRA = os.environ.get("SC_EXTRA", "").split()
RUNS = os.environ.get("SC_RUNS", os.path.join(ROOT, "runs"))

# 게임 파일이 없으면 여기서 친절하게 멈춘다 (안 그러면 C++ 이 알 수 없는 예외로 죽는다)
def _check_files():
    missing = [f for f in ("StarDat.mpq", "BrooDat.mpq", "patch_rt.mpq")
               if not os.path.exists(os.path.join(MPQ, f))
               and not os.path.exists(os.path.join(MPQ, f.replace("patch_rt", "Patch_rt")))]
    if missing:
        sys.exit(f"게임 파일이 없습니다: {', '.join(missing)}\n"
                 f"  넣을 곳: {MPQ}\n"
                 f"  준비 방법: setup/게임파일.md 를 보세요 (스타크래프트 원작은 무료입니다)")
    if not os.path.exists(BIN):
        sys.exit(f"학습 환경이 빌드되지 않았습니다: {BIN}\n"
                 f"  빌드: cd engine && ./build.sh bwmicro_sp")

# 환경에 넘기는 손잡이. 이게 바뀌면 환경을 다시 띄워야 합니다.
ENV_KEYS = ["r-enemy-hp", "r-ally-hp", "r-kill", "r-death", "r-win", "r-lose",
            "r-timeout", "r-misfire", "r-step", "r-survive", "frame-skip"]
# 학습기 안에서만 쓰는 손잡이. 환경을 다시 안 띄워도 됩니다.
TRAIN_KEYS = ["ent", "gamma"]


# ── 환경 래퍼 ────────────────────────────────────────────────────────
class Env:
    """bwmicro_sp 프로세스 하나. 한 스텝에 양쪽(P0/P1) 관측이 같이 나옵니다.

    프로토콜 (계약서 그대로):
        C++ → py : 'O' + i32 ep + i32 step + f32 rew + u8 done + u8 nm + u8 nz
                       + f32[nm*39] obs + u8[nm] alive      ← P0
                   'Q' + (완전히 같은 레이아웃)              ← P1
        py → C++ : 'A' + u8[nm]   ← P0 행동
                   'B' + u8[nz]   ← P1 행동
        판 종료  : 'E' + f32 term + u8 done + u8 surv + f32 misfire   ← P0
                   'F' + (같은 레이아웃)                              ← P1
    """

    def __init__(self, seed, ally, allies, enemy, enemies, max_steps, knobs, extra=None):
        self.seed, self.max_steps = seed, max_steps
        self.extra = list(EXTRA if extra is None else extra)
        self.ally, self.allies = ally, allies
        self.enemy, self.enemies = enemy, enemies
        self.knobs = dict(knobs)
        self.finished = []      # 끝난 판들 (ret0, ret1, done, surv0, surv1, mf0, mf1, 길이)
        self.open()

    def open(self):
        cmd = [BIN, "--data", MPQ, "--map", MAP,
               "--selfplay", "--strict-fire", "--box-range",     # 항상 켭니다. 끄는 옵션은 없습니다
               "--ally", self.ally, "--enemy", self.enemy,
               "--allies", str(self.allies), "--enemies", str(self.enemies),
               "--max-steps", str(self.max_steps), "--episodes", "100000000",
               "--seed", str(self.seed)]
        for k in ENV_KEYS:
            cmd += [f"--{k}", str(int(self.knobs[k]))]
        cmd += self.extra
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, cwd=MPQ)
        self.ret0 = self.ret1 = 0.0
        self.last_step = 0
        self.h0 = self.h1 = None      # 유닛별 최근 행동 (기억 모드에서만 씀)

    def _rd(self, n):
        b = b""
        while len(b) < n:
            c = self.p.stdout.read(n - len(b))
            if not c:
                raise EOFError("환경이 끊겼습니다")
            b += c
        return b

    def _one(self, want):
        tag = self._rd(1)
        if tag != want:
            raise EOFError(f"프로토콜 어긋남: {tag!r} (기대 {want!r})")
        ep, st, rew, done, nm, nz = struct.unpack("<iifBBB", self._rd(15))
        obs = np.frombuffer(self._rd(4 * nm * RAW_OBS), dtype=np.float32).reshape(nm, RAW_OBS)
        alv = np.frombuffer(self._rd(nm), dtype=np.uint8).astype(np.float32)
        return obs.copy(), alv.copy(), rew, done, st

    def poll(self):
        """관측 한 쌍('O' 다음 'Q')을 읽습니다. (o0, a0, r0, o1, a1, r1, done)"""
        o0, a0, r0, done, st = self._one(b'O')
        o1, a1, r1, _, _ = self._one(b'Q')
        if MEM_DIM:
            if self.h0 is None:
                self.h0 = [[] for _ in range(len(o0))]
                self.h1 = [[] for _ in range(len(o1))]
            o0 = np.concatenate([o0, mem_row(st, self.h0, len(o0))], 1)
            o1 = np.concatenate([o1, mem_row(st, self.h1, len(o1))], 1)
        self.ret0 += r0
        self.ret1 += r1
        self.last_step = st
        return o0, a0, r0, o1, a1, r1, done

    def finish(self):
        """판이 끝난 뒤 'E' 와 'F' 를 **둘 다** 읽습니다.
        ★ 하나만 읽으면 C++ 이 std::exit(4) 로 즉사합니다.
        원본 ppo_micro.py 는 종료 점수를 로그에만 쓰고 학습에 안 넣었습니다.
        그러면 '이기면/지면/시간만 끌면' 손잡이가 전부 무용지물이라 여기서는 되돌려줍니다."""
        if self._rd(1) != b'E':
            raise EOFError("'E' 가 안 왔습니다")
        t0, done, sv0, mf0 = struct.unpack("<fBBf", self._rd(10))
        if self._rd(1) != b'F':
            raise EOFError("'F' 가 안 왔습니다")
        t1, _, sv1, mf1 = struct.unpack("<fBBf", self._rd(10))
        self.finished.append((self.ret0 + t0, self.ret1 + t1, done, sv0, sv1,
                              mf0, mf1, self.last_step))
        self.ret0 = self.ret1 = 0.0
        self.h0 = self.h1 = None      # 판이 바뀌면 기억도 지운다
        return t0, t1

    def act(self, a0, a1):
        # ★ 반드시 int 로 풀어서 보냅니다. numpy 배열을 bytes() 에 넣으면
        #   int64 원시버퍼가 나가서 프로토콜이 통째로 깨집니다.
        if MEM_DIM and self.h0 is not None:
            for h, aa in ((self.h0, a0), (self.h1, a1)):
                for i, x in enumerate(aa):
                    if i < len(h):
                        h[i].insert(0, int(x) % N_ACT); del h[i][MEM_K:]
        try:
            self.p.stdin.write(b'A' + bytes(int(x) % N_ACT for x in a0))
            self.p.stdin.write(b'B' + bytes(int(x) % N_ACT for x in a1))
            self.p.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise EOFError(f"행동을 못 보냈습니다: {e}")

    def close(self):
        try:
            self.p.stdin.close()
            self.p.terminate()
            self.p.wait(timeout=5)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass


# ── 정책망 ──────────────────────────────────────────────────────────
class Policy(nn.Module):
    """관측 39 → 은닉 128×2 → (행동 12, 앞으로 딸 점수 예상 1). 한쪽 팀 전원이 공유합니다."""

    def __init__(self, obs=OBS_DIM, hid=128, act=N_ACT, split_value=False, norm_value=False):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs, hid), nn.Tanh(),
                                  nn.Linear(hid, hid), nn.Tanh())
        # ★ 값 전용 몸통. 정책과 몸통을 나눠 쓰면 값 손실이 정책 특징을 흔들지 않는다.
        #   (실측: 몸통을 공유하면 값의 설명분산 EV 가 0.02~0.16 에서 안 올라간다.
        #    같은 관측으로 값만 전담하는 망을 새로 학습시키면 0.63~0.76 이 나온다.
        #    그리고 무너지는 속도가 EV 순서와 정확히 같았다 — FD 0.02 는 682판 만에 무너졌다.)
        self.split = bool(split_value)
        self.vbody = (nn.Sequential(nn.Linear(obs, hid), nn.Tanh(),
                                    nn.Linear(hid, hid), nn.Tanh()) if self.split else None)
        self.pi = nn.Linear(hid, act)
        self.v = nn.Linear(hid, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, 0.01)   # 처음엔 거의 균등하게 고릅니다
        # ★ 수익 정규화. 수익 표준편차가 26~33 인데 값머리는 tanh(±1) 위의 Linear 한 장이라
        #   그대로 맞추려면 가중치가 커지고, 그 큰 기울기가 몸통을 타고 정책까지 간다.
        #   목표를 평균0·표준편차1 로 바꿔 학습하고, 밖으로 낼 때 되돌린다.
        self.norm_value = bool(norm_value)
        # persistent=False — 체크포인트에 안 넣는다. 넣으면 예전 체크포인트가
        # "없는 키" 로 전부 적재 실패한다(스튜디오·평가 스크립트가 다 깨진다).
        # 눈금은 이어받을 때 첫 배치에서 다시 잡으므로 손해가 없다.
        self.register_buffer("ret_mu", torch.zeros(()), persistent=False)
        self.register_buffer("ret_sd", torch.ones(()), persistent=False)
        self.register_buffer("ret_ready", torch.zeros(()), persistent=False)

    def forward(self, x):
        h = self.body(x)
        hv = self.vbody(x) if self.split else h
        v = self.v(hv).squeeze(-1)
        if self.norm_value:
            v = v * self.ret_sd + self.ret_mu          # 항상 원래 눈금으로 내보낸다
        return self.pi(h), v

    def observe_returns(self, ret):
        """수익의 눈금을 천천히 따라간다. ppo() 가 배치마다 한 번 부른다."""
        if not self.norm_value:
            return
        with torch.no_grad():
            m = ret.mean(); s = ret.std().clamp_min(1e-3)
            if float(self.ret_ready) < 0.5:
                self.ret_mu.copy_(m); self.ret_sd.copy_(s); self.ret_ready.fill_(1.0)
            else:
                self.ret_mu.mul_(0.99).add_(0.01 * m)
                self.ret_sd.mul_(0.99).add_(0.01 * s)

    def load_compat(self, sd):
        """예전 체크포인트(몸통 공유·정규화 없음)도 받아준다.
        값 전용 몸통이 있으면 정책 몸통 가중치를 복사해서 출발시킨다."""
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if self.split and not any(k.startswith("vbody.") for k in sd):
            self.vbody.load_state_dict({k[len("body."):]: v
                                        for k, v in sd.items() if k.startswith("body.")})
        return missing, unexpected

    def reset_value(self):
        """판이 바뀌면 예전 점수 예상이 전부 틀려서, 그대로 두면 잘하던 행동까지 밀어냅니다."""
        nn.init.orthogonal_(self.v.weight, 1.0)
        nn.init.zeros_(self.v.bias)
        if self.split:
            for m in self.vbody.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, 1.0); nn.init.zeros_(m.bias)
        if self.norm_value:
            self.ret_ready.zero_()          # 수익 눈금도 새 판에서 다시 잡는다


# ── 한쪽 팀의 GAE / PPO. 이 두 함수를 P0 와 P1 에 각각 한 번씩 부릅니다 ──────
def gae(b_rew, b_val, b_done, last_v, gamma, lam):
    """b_rew (T,N) 팀 점수 · b_val (T,N,M) · b_done (T,N) → (이득, 목표값)"""
    T, N, M = b_val.shape
    adv = np.zeros((T, N, M), np.float32)
    run = np.zeros((N, M), np.float32)
    for t in reversed(range(T)):
        nonterm = 1.0 - b_done[t][:, None]
        nv = last_v if t == T - 1 else b_val[t + 1]
        delta = b_rew[t][:, None] + gamma * nv * nonterm - b_val[t]
        run = delta + gamma * lam * nonterm * run
        adv[t] = run
    return adv, adv + b_val


# ★ 선생 닻: 베낀 망에서 이어 학습하면 PPO 가 즉시 밀어낸다(홀드 75% → 어택땅 87%, 4세대 만에, ZB6 되돌림 8회).
#   이득의 즉시 항(피해 보상)이 지연 항(승리)을 이기기 때문. 시작 망을 얼려 두고 KL(닻‖정책) 을 손실에 더해
#   좋은 자리에서 못 떠나게 붙든다. 이득이 확실한 방향으로는 여전히 움직인다.
ANCHOR = None
def ppo(net, opt, b_obs, b_act, b_logp, adv, ret, b_alv, args, dev, pol_on=True):
    """살아있는 유닛만 학습에 씁니다. 죽은 슬롯은 관측이 0 이라 넣으면 독입니다."""
    mask = b_alv.reshape(-1) > 0.5
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0.0
    fo = torch.as_tensor(b_obs.reshape(-1, OBS_DIM)[mask], device=dev)
    fa = torch.as_tensor(b_act.reshape(-1)[mask], device=dev)
    fl = torch.as_tensor(b_logp.reshape(-1)[mask], device=dev)
    fadv = torch.as_tensor(adv.reshape(-1)[mask], device=dev)
    fret = torch.as_tensor(ret.reshape(-1)[mask], device=dev)
    fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
    net.observe_returns(fret)
    last_ent, last_vl = 0.0, 0.0
    # ★ 예열 중에는 몸통까지 진짜로 얼린다.
    #   값머리와 정책이 몸통(Linear 2장)을 공유하기 때문에, 값 손실만 줘도 몸통이 움직여서
    #   정책이 같이 망가진다. 실제로 "정책을 껐는데" 탐험이 0.08 → 2.26 으로 튀었다.
    if not pol_on:
        for q in net.body.parameters(): q.requires_grad_(False)
        for q in net.pi.parameters():   q.requires_grad_(False)
    for _ in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, args.minibatch):
            idx = perm[s:s + args.minibatch]
            logits, v = net(fo[idx])
            dist = torch.distributions.Categorical(logits=logits)
            lp = dist.log_prob(fa[idx])
            ratio = (lp - fl[idx]).exp()
            a1 = ratio * fadv[idx]
            a2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * fadv[idx]
            pl = -torch.min(a1, a2).mean()
            if net.norm_value:
                # 값은 원래 눈금으로 나오므로, 손실만 정규화된 자리에서 잰다.
                vl = F.mse_loss((v - net.ret_mu) / net.ret_sd,
                                (fret[idx] - net.ret_mu) / net.ret_sd)
            else:
                vl = F.mse_loss(v, fret[idx])
            ent = dist.entropy().mean()
            # ★ 값머리 예열: 베끼기로 시작할 때는 값머리가 그 정책의 실력을 아직 모른다.
            #   그대로 학습을 붙이면 이득 추정이 전부 크게 음수로 나와서
            #   "내가 하는 건 다 나쁘다" 가 되고, 잘 베낀 정책이 몇천 판 만에 균등분포로 무너진다.
            #   (실제로 두 번 그렇게 부서졌다) 그래서 처음 얼마간은 값만 맞춘다.
            loss = (pl if pol_on else pl.detach() * 0) + args.vf * vl - (args.ent * ent if pol_on else 0)
            if ANCHOR is not None and args.anchor_kl > 0 and pol_on:
                with torch.no_grad():
                    ref_logits, _ = ANCHOR(fo[idx])
                    ref_lp = F.log_softmax(ref_logits, -1)
                kl = (ref_lp.exp() * (ref_lp - F.log_softmax(logits, -1))).sum(-1).mean()
                loss = loss + args.anchor_kl * kl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            last_ent, last_vl = float(ent.item()), float(vl.item())
    if not pol_on:
        for q in net.parameters(): q.requires_grad_(True)
    return last_ent, last_vl


def move_px(prev_o, cur_o, prev_a, cur_a):
    """스텝당 팀 평균 이동 픽셀. 관측의 자기 절대좌표 o[3],o[4] 는 (pos-C)/512 라
    차이에 512 를 곱하면 픽셀입니다. ★ '둘 다 서서 자동교전' 을 잡아내는 지표입니다."""
    m = (prev_a > 0.5) & (cur_a > 0.5)
    k = int(m.sum())
    if k == 0:
        return 0.0, 0
    d = np.hypot(cur_o[m, 3] - prev_o[m, 3], cur_o[m, 4] - prev_o[m, 4]) * 512.0
    return float(d.sum()), k


def top_desc(hist):
    k = int(np.argmax(hist))
    return f"{ACT_NAME[k]} {hist[k]*100:.0f}%"


# ── 손코딩 정책 (배선 검사에서만 씁니다. preflight.py 의 POL 을 그대로 옮겼습니다) ──
DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def dir_action(dx, dy):
    n = float(np.hypot(dx, dy))
    if n < 1e-6:
        return 0
    dx, dy = dx / n, dy / n
    best, bi = -9.0, 3
    for i, (ax, ay) in enumerate(DIRS):
        an = float(np.hypot(ax, ay))
        d = (dx * ax + dy * ay) / an
        if d > best:
            best, bi = d, i + 1
    return bi


def p_kite(o, alive):    # 무빙샷: 쿨다운 중이면 물러납니다
    a = []
    for i in range(len(o)):
        if alive[i] < 0.5:
            a.append(0); continue
        ex, ey, inr, cd = o[i][5] * 256.0, o[i][6] * 256.0, o[i][10] > 0.5, o[i][1]
        a.append(9 if (inr and cd < 0.1) else (11 if inr else dir_action(ex, ey)))
    return a


def p_rush(o, alive):    # 돌진: 사거리 안이면 쏘고 아니면 붙습니다
    a = []
    for i in range(len(o)):
        if alive[i] < 0.5:
            a.append(0); continue
        ex, ey, inr = o[i][5] * 256.0, o[i][6] * 256.0, o[i][10] > 0.5
        a.append(9 if inr else dir_action(ex, ey))
    return a


# ── 배선 검사 ───────────────────────────────────────────────────────
def selftest(args):
    """학습을 안 합니다. 진짜 bwmicro_sp 를 손코딩 정책으로 굴려 경험만 모은 뒤,
    P0 에만 상수 +1, P1 에만 상수 -1 을 넣고 한 번 업데이트합니다.
    두 망의 '앞으로 딸 점수 예상' 부호가 반대로 갈라져야 배선이 맞는 것입니다.
    (부호를 뒤집어서도 한 번 더 봅니다. 한쪽 점수를 양쪽에 쓰는 실수를 잡습니다.)"""
    dev = torch.device("cpu")
    N, T = 2, 48
    A, B = 1, 6
    knobs = {k: DEFAULT_KNOBS[k] for k in ENV_KEYS}
    knobs["frame-skip"] = 4
    print("배선 검사 — 벌처1 vs 저글링6, 환경 2개 × 48스텝, 손코딩(무빙샷 vs 돌진)")
    envs = [Env(777 + i * 1000, "vulture", A, "zergling", B, 200, knobs) for i in range(N)]
    obs0 = np.zeros((N, A, OBS_DIM), np.float32); alv0 = np.zeros((N, A), np.float32)
    obs1 = np.zeros((N, B, OBS_DIM), np.float32); alv1 = np.zeros((N, B), np.float32)
    for i, e in enumerate(envs):
        o0, a0, _, o1, a1, _, _ = e.poll()
        obs0[i], alv0[i], obs1[i], alv1[i] = o0, a0, o1, a1

    bo0 = np.zeros((T, N, A, OBS_DIM), np.float32); ba0 = np.zeros((T, N, A), np.int64)
    bv0 = np.zeros((T, N, A), np.float32); bl0 = np.zeros((T, N, A), np.float32)
    ma0 = np.zeros((T, N, A), np.float32)
    bo1 = np.zeros((T, N, B, OBS_DIM), np.float32); ba1 = np.zeros((T, N, B), np.int64)
    bv1 = np.zeros((T, N, B), np.float32); bl1 = np.zeros((T, N, B), np.float32)
    ma1 = np.zeros((T, N, B), np.float32)
    b_done = np.zeros((T, N), np.float32)

    for t in range(T):
        act0 = np.array([p_kite(obs0[i], alv0[i]) for i in range(N)], np.int64)
        act1 = np.array([p_rush(obs1[i], alv1[i]) for i in range(N)], np.int64)
        bo0[t], ma0[t], ba0[t] = obs0, alv0, act0
        bo1[t], ma1[t], ba1[t] = obs1, alv1, act1
        for i, e in enumerate(envs):
            e.act(act0[i], act1[i])
        for i, e in enumerate(envs):
            o0, a0, _, o1, a1, _, d = e.poll()
            b_done[t, i] = 1.0 if d else 0.0
            if d:
                e.finish()
                o0, a0, _, o1, a1, _, _ = e.poll()
            obs0[i], alv0[i], obs1[i], alv1[i] = o0, a0, o1, a1
    for e in envs:
        e.close()

    # 관측이 정말 서로 다른 배열인가 (같은 걸 두 번 담았으면 여기서 걸립니다)
    assert bo0.shape[2] == A and bo1.shape[2] == B, "슬롯 수가 뒤바뀌었습니다"
    assert not np.allclose(bo0[:, :, 0, :5], bo1[:, :, 0, :5]), "P0 와 P1 관측이 같습니다"
    print(f"  관측 모양 P0{tuple(bo0.shape)} P1{tuple(bo1.shape)} · 서로 다름 확인")

    class A2:  # 업데이트용 손잡이
        epochs, minibatch, clip, vf, ent = 10, 8192, 0.2, 0.5, 0.0
    ok = True
    for sign, label in ((+1.0, "P0=+1 / P1=-1"), (-1.0, "P0=-1 / P1=+1")):
        torch.manual_seed(1234)
        n0, n1 = Policy().to(dev), Policy().to(dev)
        o0 = torch.optim.Adam(n0.parameters(), lr=1e-3, eps=1e-5)
        o1 = torch.optim.Adam(n1.parameters(), lr=1e-3, eps=1e-5)
        with torch.no_grad():
            g0, v0 = n0(torch.as_tensor(bo0.reshape(-1, OBS_DIM)))
            g1, v1 = n1(torch.as_tensor(bo1.reshape(-1, OBS_DIM)))
            lp0 = torch.distributions.Categorical(logits=g0).log_prob(
                torch.as_tensor(ba0.reshape(-1)))
            lp1 = torch.distributions.Categorical(logits=g1).log_prob(
                torch.as_tensor(ba1.reshape(-1)))
        bv0[:] = v0.numpy().reshape(T, N, A); bl0[:] = lp0.numpy().reshape(T, N, A)
        bv1[:] = v1.numpy().reshape(T, N, B); bl1[:] = lp1.numpy().reshape(T, N, B)
        # ★ 인위적 점수 주입 — 여기가 검사의 핵심입니다
        r0 = np.full((T, N), +1.0 * sign, np.float32)
        r1 = np.full((T, N), -1.0 * sign, np.float32)
        ad0, rt0 = gae(r0, bv0, b_done, bv0[-1], 0.99, 0.95)
        ad1, rt1 = gae(r1, bv1, b_done, bv1[-1], 0.99, 0.95)
        ppo(n0, o0, bo0, ba0, bl0, ad0, rt0, ma0, A2, dev)
        ppo(n1, o1, bo1, ba1, bl1, ad1, rt1, ma1, A2, dev)
        with torch.no_grad():
            m0 = ma0.reshape(-1) > 0.5
            m1 = ma1.reshape(-1) > 0.5
            _, q0 = n0(torch.as_tensor(bo0.reshape(-1, OBS_DIM)[m0]))
            _, q1 = n1(torch.as_tensor(bo1.reshape(-1, OBS_DIM)[m1]))
            e0, e1 = float(q0.mean()), float(q1.mean())
        good = (e0 > 0 > e1) if sign > 0 else (e1 > 0 > e0)
        ok = ok and good
        print(f"  {label:<16} → P0 점수예상 {e0:+.3f} · P1 점수예상 {e1:+.3f}  "
              f"{'통과' if good else '실패'}")
        assert good, ("배선이 틀렸습니다. P0 와 P1 의 점수가 섞이고 있습니다 "
                      f"(P0 {e0:+.3f}, P1 {e1:+.3f})")
    print("배선 검사 통과 — P0 와 P1 의 점수·관측·마스크가 안 섞입니다." if ok else "실패")
    return 0


# ── 손잡이 기본값 (계약서 표 그대로) ───────────────────────────────────
DEFAULT_KNOBS = {"r-enemy-hp": 300, "r-ally-hp": 30, "r-kill": 200, "r-death": 100,
                 "r-win": 1000, "r-lose": 1000, "r-timeout": 2500, "r-misfire": 0,
                 "r-step": 1, "r-survive": 0, "frame-skip": 4, "ent": 0.01, "gamma": 0.999}


def read_control(path):
    """studio.py 가 쓴 control.json 을 읽습니다. 읽기만 합니다 — 절대 안 씁니다.
    쓰는 도중에 읽으면 깨진 JSON 이 나오므로 조용히 무시합니다."""
    try:
        with open(path, "r") as f:
            d = json.load(f)
    except Exception:
        return {}, None
    if not isinstance(d, dict):
        return {}, None
    src = {}
    for holder in ("sliders", "patch", "train"):
        if isinstance(d.get(holder), dict):
            src.update(d[holder])
    src.update({k: v for k, v in d.items() if k in ENV_KEYS or k in TRAIN_KEYS})
    knobs = {k: v for k, v in src.items()
             if (k in ENV_KEYS or k in TRAIN_KEYS) and isinstance(v, (int, float))}
    cmd = d.get("cmd") or d.get("state") or d.get("action")
    return knobs, (str(cmd) if cmd else None)


def main():
    ap = argparse.ArgumentParser(description="비대칭 셀프플레이 PPO 학습기")
    # 판 구성
    ap.add_argument("--ally", type=str, default="vulture")
    ap.add_argument("--allies", type=int, default=1)
    ap.add_argument("--enemy", type=str, default="zergling")
    ap.add_argument("--enemies", type=int, default=6)
    ap.add_argument("--frame-skip", type=int, default=DEFAULT_KNOBS["frame-skip"],
                    help="몇 프레임마다 한 번 명령할지. 그대로 APM 입니다")
    ap.add_argument("--max-steps", type=int, default=350)
    # 점수 손잡이 (단위 1/100. 계약서 표의 기본값)
    ap.add_argument("--r-enemy-hp", type=int, default=DEFAULT_KNOBS["r-enemy-hp"])
    ap.add_argument("--r-ally-hp", type=int, default=DEFAULT_KNOBS["r-ally-hp"])
    ap.add_argument("--r-kill", type=int, default=DEFAULT_KNOBS["r-kill"])
    ap.add_argument("--r-death", type=int, default=DEFAULT_KNOBS["r-death"])
    ap.add_argument("--r-win", type=int, default=DEFAULT_KNOBS["r-win"])
    ap.add_argument("--r-lose", type=int, default=DEFAULT_KNOBS["r-lose"])
    ap.add_argument("--r-timeout", type=int, default=DEFAULT_KNOBS["r-timeout"])
    ap.add_argument("--r-misfire", type=int, default=DEFAULT_KNOBS["r-misfire"])
    ap.add_argument("--r-step", type=int, default=DEFAULT_KNOBS["r-step"])
    # ★ 이긴 판에서 살아남은 아군 한 기당 보너스 — '이기되 안 잃고 이기기'
    ap.add_argument("--r-survive", type=int, default=DEFAULT_KNOBS["r-survive"])
    # 학습 손잡이
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=DEFAULT_KNOBS["ent"])
    ap.add_argument("--gamma", type=float, default=DEFAULT_KNOBS["gamma"])
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--curriculum-key", type=str, default="--mines",
                    help="커리큘럼이 갈아끼울 환경 옵션 이름. 마인판은 --mines, bwctrl 판은 --enemies-spawn "
                         "(--enemies 는 관측 슬롯 수라 고정하고, 실제 스폰 수만 바꾼다)")
    ap.add_argument("--stage-eval", type=int, default=0,
                    help="0보다 크면 단계 승격을 학습 로그 승률이 아니라 '굳힘(argmax) 평가 N판' 으로 판정한다. "
                         "★ 학습 로그 승률은 표본추출값이라 실력보다 낮다(질럿 판에서 26%% 실력이 3%%로 찍혔다)")
    ap.add_argument("--stage-eval-every", type=int, default=100,
                    help="굳힘 평가를 몇 업데이트마다 돌리나 (최소 판수를 채운 뒤부터)")
    ap.add_argument("--stage-guard", type=float, default=0.0,
                    help="0보다 크면 단계 안에서 굳힘 최고 세대를 들고 있다가 그보다 이만큼(승률 비율) 떨어지면 "
                         "그 세대로 되돌리고 옵티마이저를 새로 만든다. 최소 판수 전에도 굳힘을 잰다 "
                         "(투혼 마린 14→15 승격 직후 160판 만에 100%%→10%% 붕괴 실측)")
    ap.add_argument("--promote-next-min", type=float, default=0.0,
                    help="0보다 크면 승격 전에 다음 계단을 미리 굳힘으로 재서 이 승률 이상일 때만 올린다. "
                         "★ 이긴 판이 하나도 안 나오는 계단으로 올라가면 배울 신호가 없어 굳는다 "
                         "(저글링 vs 리버 3: 38%% 한 번 튄 순간 승격 → 230만 판 0%%)")
    ap.add_argument("--reset-value-on-stage", action="store_true",
                    help="단계를 넘길 때 값머리를 새로 시작한다. 판이 바뀌면 예전 값 추정이 전부 틀려서 "
                         "잘하던 행동까지 밀어낸다 (럴커 편 실측)")
    ap.add_argument("--curriculum", type=str, default="",
                    help="쉬운 판부터 올린다. 예 1:60:15000,2:60:30000,4:0:0 "
                         "= 마인1개로 시작해 승률60%% AND 최소15000판이면 2개로. "
                         "★ 승률만 걸면 안 된다 - 지난번에 28판 만에 통과해버렸다")
    ap.add_argument("--out", type=str, default="sp1")
    # 겉으로 안 내놓는 것들
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--split-value", action="store_true",
                    help="값 전용 몸통을 따로 둔다. 정책과 몸통을 공유하면 값이 안 배워진다(EV 0.02~0.16).")
    ap.add_argument("--norm-value", action="store_true",
                    help="값 목표를 평균0·표준편차1 로 정규화해서 학습한다(수익 표준편차가 26~33 이라 필요하다).")
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--behavior-thresh", type=float, default=0.30,
                    help="행동 분포가 이만큼(L1) 바뀌면 세대를 저장합니다")
    ap.add_argument("--vf-warmup", type=int, default=0,
                   help="처음 N 업데이트는 값머리만 맞춘다(정책은 얼려둔다). "
                        "베끼기로 시작할 때 필수 — 안 넣으면 잘 베낀 정책이 바로 무너진다.")
    ap.add_argument("--memory", type=str, default="none",
                   choices=["none", "clock", "acts", "both"],
                   help="망에게 기억을 준다. clock=박자 위상(사인/코사인 4쌍), "
                        "acts=최근 5개 행동, both=둘 다. 화면만으로는 택견을 담을 수 없다.")
    ap.add_argument("--anchor-kl", type=float, default=0.0,
                    help="선생 닻 세기. --init-from 망을 얼려 두고 KL(닻‖정책)×이 값을 손실에 더한다 (베낀 망 이어 학습용)")
    ap.add_argument("--init-sharpen", type=float, default=1.0,
                    help="이어받은 망의 정책 머리(pi) 가중치·편향에 이 배수를 곱해 표본추출을 날카롭게 한다 "
                         "(argmax 는 그대로). 굳힘 93%%인 망이 표본추출로는 22%%만 이겨서 PPO 가 지는 판만 보는 문제")
    ap.add_argument("--init-from", type=str, default="",
                   help="P0 망을 이 체크포인트에서 시작합니다 (맨땅 말고 이어서 배우기). "
                        "★ 사람 조건처럼 맨땅에서는 한 판도 못 이기는 판을 배울 때 씁니다.")
    ap.add_argument("--p1-script", type=str, default="none", choices=["none", "hold", "amove"],
                    help="P1(적)을 학습시키지 않고 고정 대본으로 움직인다. hold=홀드(제자리 자동공격), amove=어택땅. "
                         "리버처럼 '서 있는 적' 판에서 P0 만 배우게 할 때 쓴다.")
    ap.add_argument("--selftest", action="store_true", help="배선 검사만 하고 끝냅니다")
    args = ap.parse_args()

    if args.memory != "none":
        set_memory(args.memory)
        print(f"기억 {args.memory} — 관측 {RAW_OBS} → {OBS_DIM}")
    if args.selftest:
        return selftest(args)

    # 결과 폴더: 절대경로면 그대로, 아니면 runs_sp 아래
    o = os.path.expanduser(args.out)
    out = o if os.path.isabs(o) else os.path.join(RUNS, o)
    os.makedirs(os.path.join(out, "ckpt"), exist_ok=True)
    mpath = os.path.join(out, "metrics.jsonl")
    cpath = os.path.join(out, "control.json")

    ep_done = 0

    def emit(row):
        """metrics.jsonl 에 한 줄 붙입니다. ★ 통짜 덮어쓰기 금지 —
        기존 log.json 방식은 vult1 런에서 885초 중 393초를 먹었습니다."""
        with open(mpath, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    mk = lambda: Policy(OBS_DIM, split_value=args.split_value, norm_value=args.norm_value)
    net0, net1 = mk().to(dev), mk().to(dev)   # P0=벌처망, P1=저글링망
    if args.split_value or args.norm_value:
        print(f"   값머리: {'전용 몸통' if args.split_value else '몸통 공유'}"
              f" · {'수익 정규화' if args.norm_value else '정규화 없음'}")
    # ★ 이어서 배우기. 맨땅에서 한 판도 못 이기는 판은 보상이 평평해서 학습이 시작조차 안 된다.
    #   (사람 조건 25만 판에서 승리 0회 — 점수가 −24 언저리에서 25만 판 내내 안 움직였다)
    #   그럴 땐 쉬운 조건에서 배운 망을 가져다 놓고 거기서 다시 배우게 한다.
    if args.init_from:
        ck = torch.load(os.path.expanduser(args.init_from), map_location=dev)
        net0.load_compat(ck["model"])
        print(f"P0 망을 이어받았습니다: {args.init_from} "
              f"(세대 {ck.get('gen')} · {ck.get('ep', 0):,}판)")
        if args.anchor_kl > 0:
            global ANCHOR
            ANCHOR = copy.deepcopy(net0).to(dev).eval()
            for q in ANCHOR.parameters(): q.requires_grad_(False)
            print(f"   선생 닻 고정 (KL 세기 {args.anchor_kl:g})")
        if args.init_sharpen != 1.0:
            with torch.no_grad():
                net0.pi.weight.mul_(args.init_sharpen); net0.pi.bias.mul_(args.init_sharpen)
            print(f"   정책 머리 {args.init_sharpen:g}배 날카롭게 (표본추출이 굳힘에 가깝게)")
    opt0 = torch.optim.Adam(net0.parameters(), lr=args.lr, eps=1e-5)
    opt1 = torch.optim.Adam(net1.parameters(), lr=args.lr, eps=1e-5)

    knobs = {k: getattr(args, k.replace("-", "_")) for k in ENV_KEYS}
    N, A, B, T = args.envs, args.allies, args.enemies, args.rollout
    print(f"{args.ally} {A}기(P0) vs {args.enemy} {B}마리(P1) · 판단주기 {args.frame_skip}프레임 · "
          f"환경 {N}개 · 장치 {dev}")
    print(f"결과 폴더 {out}")

    gen = [0]

    def save_gen(why, h0, h1):
        g = gen[0]
        p0, p1 = f"ckpt/v{g:03d}.pt", f"ckpt/z{g:03d}.pt"
        torch.save({"model": net0.state_dict(), "gen": g, "ep": ep_done, "side": args.ally},
                   os.path.join(out, p0))
        torch.save({"model": net1.state_dict(), "gen": g, "ep": ep_done, "side": args.enemy},
                   os.path.join(out, p1))
        emit({"t": "gen", "gen": g, "ep": ep_done, "side": "both", "p0": p0, "p1": p1,
              "why": why, "top0": top_desc(h0), "top1": top_desc(h1)})
        print(f"   ★ 세대 {g:03d} 저장 ({why}) · P0 {top_desc(h0)} · P1 {top_desc(h1)}")
        gen[0] += 1

    stop = {"flag": False}

    def on_term(signum, frame):
        stop["flag"] = True
        print("\n멈추라는 신호를 받았습니다. 마지막 세대를 저장하고 끝냅니다.")
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    # ── 커리큘럼: 마인 개수를 단계로 올린다 ───────────────────────────
    stages = []
    if args.curriculum:
        for part in args.curriculum.split(","):
            bits = part.split(":")
            stages.append((int(bits[0]), float(bits[1]) / 100.0,
                           int(bits[2]) if len(bits) > 2 else 0))
    def extra_for(mines):
        """EXTRA 에서 커리큘럼 키(--mines / --enemies-spawn ...) 값만 갈아끼운다."""
        e = list(EXTRA)
        if mines is None: return e
        key = args.curriculum_key
        if key in e:
            e[e.index(key) + 1] = str(mines)
        else:
            e += [key, str(mines)]
        return e

    def greedy_eval(net, extra, games, seed_off=0):
        """★ 승격 판정용 굳힘 평가. 학습 환경과 별개로 환경 2개를 잠깐 띄워 argmax 로 N판 돈다.
        표본추출 승률(학습 로그)은 실력의 하한일 뿐이라 이걸로 승격하면 영영 못 넘길 수 있다."""
        # ★ seed_off 는 '평가 회차' 다. 회차마다 다른 판을 돌려야 한다 —
        #   고정 씨앗으로 재면 그 40판에만 맞춰진 세대가 80% 로 보이고
        #   새 씨앗에선 28% 였다 (마인밭 v18·v3737 실측, 2026-09-07).
        #   C++ 는 판마다 base_seed+episode 를 쓰므로(bwdragoon.cpp:673)
        #   회차 간격을 환경 간격(1000)+판수보다 크게 벌려야 지난 회차와 안 겹친다.
        base = args.seed + 90001 + seed_off * 4096
        ev = [Env(base + i * 1000, args.ally, A, args.enemy, B,
                  args.max_steps, knobs, extra=extra) for i in range(2)]
        won = tot = 0
        try:
            cur = [e.poll() for e in ev]
            while tot < games:
                for i, e in enumerate(ev):
                    o0, a0, r0, o1, a1, r1, dn = cur[i]
                    if dn:
                        e.finish(); tot += 1; won += (e.finished[-1][2] == 1)
                        cur[i] = e.poll(); continue
                    with torch.no_grad():
                        lg, _ = net(torch.as_tensor(o0, device=dev))
                        act = lg.argmax(-1).cpu().numpy()
                    e.act(act, np.zeros(len(o1), np.int64))
                    cur[i] = e.poll()
        finally:
            for e in ev: e.close()
        return won / max(1, tot), tot

    stage = 0
    cur_mines = stages[0][0] if stages else None
    stage_start_ep = 0
    # make_envs 가 cur_mines 를 읽어야 해서 참조를 하나 둔다
    _cur = {"mines": cur_mines}
    # ★ 되돌림 가드: 이 단계에서 굳힘이 제일 좋았던 세대의 가중치
    guard = {"best": -1.0, "sd": None, "n": 0}
    def guard_set(gw, gn, stage_n, why):
        """최고 세대를 갈아끼우고 파일로도 남긴다. ★ 예전엔 메모리에만 있어서
        되돌림 대상이 무엇인지 나중에 확인할 수 없었다 (HR1: 54% 상태가 어느 파일에도 없었다)."""
        guard["best"] = gw; guard["sd"] = copy.deepcopy(net0.state_dict())
        torch.save({"model": guard["sd"], "gen": "best", "wr": gw, "n": gn, "stage": stage_n, "why": why},
                   os.path.join(out, "ckpt", f"best_stage{stage_n}.pt"))
        print(f"   ★ 가드 최고 {gw*100:.0f}% ({gn}판) → ckpt/best_stage{stage_n}.pt [{why}]", flush=True)
    if stages and args.stage_guard > 0 and args.stage_eval > 0:
        gw, gn = greedy_eval(net0, extra_for(cur_mines), args.stage_eval)
        guard_set(gw, gn, cur_mines, "시작 기준")
        print(f"   [굳힘 평가] 단계 {cur_mines} 시작 기준 · {gn}판 · 승률 {gw*100:.0f}%", flush=True)

    def make_envs():
        ex = extra_for(_cur["mines"])
        return [Env(args.seed + i * 1000, args.ally, A, args.enemy, B, args.max_steps,
                    knobs, extra=ex)
                for i in range(N)]

    envs = make_envs()
    if stages:
        print("커리큘럼: " + " → ".join(
            f"{args.curriculum_key}={m}(승률{int(t*100)}%+{e:,}판)" for m, t, e in stages), flush=True)
    obs0 = np.zeros((N, A, OBS_DIM), np.float32); alv0 = np.zeros((N, A), np.float32)
    obs1 = np.zeros((N, B, OBS_DIM), np.float32); alv1 = np.zeros((N, B), np.float32)
    for i, e in enumerate(envs):
        a, b, _, c, d, _, _ = e.poll()
        obs0[i], alv0[i], obs1[i], alv1[i] = a, b, c, d

    def restart(i, msg):
        """환경 하나가 죽었을 때. ★ 원본에는 EOFError 를 잡는 곳이 아예 없어서
        환경 하나가 죽으면 학습이 통째로 죽습니다."""
        nonlocal envs
        try:
            envs[i].close()
        except Exception:
            pass
        envs[i] = Env(args.seed + i * 1000 + 7, args.ally, A, args.enemy, B,
                      args.max_steps, knobs)
        a, b, _, c, d, _, _ = envs[i].poll()
        obs0[i], alv0[i], obs1[i], alv1[i] = a, b, c, d
        emit({"t": "err", "ep": ep_done, "msg": f"환경 {i}번이 죽어서 재기동했습니다 ({msg})"})
        print(f"   ! 환경 {i}번 재기동 ({msg})")

    hist0 = hist1 = sm0 = sm1 = None
    recent = collections.deque(maxlen=max(20, 8 * N))   # 최근 끝난 판들
    t_start = time.time()
    save_gen("시작", np.ones(N_ACT) / N_ACT, np.ones(N_ACT) / N_ACT)   # 0세대 = 학습 전
    applied = dict(knobs); applied["ent"] = args.ent; applied["gamma"] = args.gamma

    for upd in range(args.updates):
        if stop["flag"]:
            break
        # ── 손잡이 확인 (업데이트 경계에서만) ──
        pending, cmd = read_control(cpath)
        while cmd in ("pause", "paused", "멈춤") and not stop["flag"]:
            time.sleep(0.3)
            pending, cmd = read_control(cpath)
        if cmd in ("stop", "stopped", "정지"):
            stop["flag"] = True
            break
        patch = {k: v for k, v in pending.items() if applied.get(k) != v}
        if patch:
            env_changed = any(k in ENV_KEYS for k in patch)
            for k, v in patch.items():
                applied[k] = v
                if k in ENV_KEYS:
                    knobs[k] = int(v)
            args.ent = float(applied["ent"]); args.gamma = float(applied["gamma"])
            if env_changed:
                # 판이 바뀌면 예전 점수 예상이 전부 틀립니다. 그대로 두면 잘하던 행동까지 밀어냅니다.
                net0.reset_value(); net1.reset_value()
                for e in envs:
                    e.close()
                envs = make_envs()
                for i, e in enumerate(envs):
                    a, b, _, c, d, _, _ = e.poll()
                    obs0[i], alv0[i], obs1[i], alv1[i] = a, b, c, d
            emit({"t": "cfg", "ep": ep_done, "patch": patch, "reset_value": bool(env_changed)})
            print(f"   ● 손잡이 반영 {patch}" + (" (점수 예상 초기화)" if env_changed else ""))

        # ── 롤아웃 ──
        bo0 = np.zeros((T, N, A, OBS_DIM), np.float32); ba0 = np.zeros((T, N, A), np.int64)
        bl0 = np.zeros((T, N, A), np.float32); bv0 = np.zeros((T, N, A), np.float32)
        ma0 = np.zeros((T, N, A), np.float32); br0 = np.zeros((T, N), np.float32)
        bo1 = np.zeros((T, N, B, OBS_DIM), np.float32); ba1 = np.zeros((T, N, B), np.int64)
        bl1 = np.zeros((T, N, B), np.float32); bv1 = np.zeros((T, N, B), np.float32)
        ma1 = np.zeros((T, N, B), np.float32); br1 = np.zeros((T, N), np.float32)
        b_done = np.zeros((T, N), np.float32)

        h0 = np.zeros(N_ACT, np.int64); h1 = np.zeros(N_ACT, np.int64)
        mv0 = mv1 = 0.0; mvn0 = mvn1 = 0
        ent0_sum = ent1_sum = 0.0; ent_n = 0
        broke = False

        for t in range(T):
            if stop["flag"]:
                broke = True
                break
            with torch.no_grad():
                g0, v0 = net0(torch.as_tensor(obs0.reshape(-1, OBS_DIM), device=dev))
                d0 = torch.distributions.Categorical(logits=g0)
                s0 = d0.sample(); p0 = d0.log_prob(s0)
                g1, v1 = net1(torch.as_tensor(obs1.reshape(-1, OBS_DIM), device=dev))
                d1 = torch.distributions.Categorical(logits=g1)
                s1 = d1.sample(); p1 = d1.log_prob(s1)
                ent0_sum += float(d0.entropy().mean()); ent1_sum += float(d1.entropy().mean())
                ent_n += 1
            a0 = s0.cpu().numpy().reshape(N, A); a1 = s1.cpu().numpy().reshape(N, B)
            if args.p1_script != "none":                       # ★ 적은 대본대로: 13=홀드, 12=어택땅
                a1[:] = 13 if args.p1_script == "hold" else 12
            h0 += np.bincount(a0.reshape(-1)[alv0.reshape(-1) > 0.5], minlength=N_ACT)
            h1 += np.bincount(a1.reshape(-1)[alv1.reshape(-1) > 0.5], minlength=N_ACT)
            bo0[t], ma0[t], ba0[t] = obs0, alv0, a0
            bl0[t] = p0.cpu().numpy().reshape(N, A); bv0[t] = v0.cpu().numpy().reshape(N, A)
            bo1[t], ma1[t], ba1[t] = obs1, alv1, a1
            bl1[t] = p1.cpu().numpy().reshape(N, B); bv1[t] = v1.cpu().numpy().reshape(N, B)

            dead = []
            for i, e in enumerate(envs):
                try:
                    e.act(a0[i], a1[i])
                except EOFError as ex:
                    dead.append((i, str(ex)))
            for i, e in enumerate(envs):
                if any(i == j for j, _ in dead):
                    continue
                try:
                    o0, k0, r0, o1, k1, r1, dn = e.poll()
                    br0[t, i] = r0; br1[t, i] = r1
                    b_done[t, i] = 1.0 if dn else 0.0
                    s, c = move_px(obs0[i], o0, alv0[i], k0); mv0 += s; mvn0 += c
                    s, c = move_px(obs1[i], o1, alv1[i], k1); mv1 += s; mvn1 += c
                    if dn:
                        # 이기면/지면/시간만 끌면 점수는 'E'/'F' 로 옵니다. 학습에 넣습니다.
                        term0, term1 = e.finish()
                        br0[t, i] += term0; br1[t, i] += term1
                        o0, k0, _, o1, k1, _, _ = e.poll()     # 다음 판 첫 관측
                    obs0[i], alv0[i], obs1[i], alv1[i] = o0, k0, o1, k1
                except EOFError as ex:
                    dead.append((i, str(ex)))
            for i, msg in dead:
                b_done[t, i] = 1.0        # 끊긴 자리에서 이득 계산을 끊습니다
                br0[t, i] = br1[t, i] = 0.0
                restart(i, msg)

        if broke:
            break

        with torch.no_grad():
            _, lv0 = net0(torch.as_tensor(obs0.reshape(-1, OBS_DIM), device=dev))
            _, lv1 = net1(torch.as_tensor(obs1.reshape(-1, OBS_DIM), device=dev))
        lv0 = lv0.cpu().numpy().reshape(N, A); lv1 = lv1.cpu().numpy().reshape(N, B)

        # ★ 여기서부터 쪽별로 완전히 따로 갑니다. 섞이면 학습이 멀쩡히 돌면서 아무것도 안 늡니다.
        ad0, rt0 = gae(br0, bv0, b_done, lv0, args.gamma, args.lam)
        ad1, rt1 = gae(br1, bv1, b_done, lv1, args.gamma, args.lam)
        pol_on = (upd >= args.vf_warmup)
        # ★ 값머리가 쓸모있나 — 설명분산 EV = 1 - Var(목표-예측)/Var(목표).
        #   0 이면 '그냥 평균 찍기' 와 같다. 그러면 PPO 의 이득은 거의 전부 잡음이고,
        #   잘하는 정책을 넣어놔도 무작위 보행으로 밀려난다.
        #   (실측: EV 0.02 인 FD 는 682판, 0.16 인 택견은 6,579판 만에 무너졌다)
        def _ev(bv, rt, ma):
            m = ma.reshape(-1) > 0.5
            if m.sum() < 8: return float("nan")
            t = rt.reshape(-1)[m].astype(np.float64)
            q = bv.reshape(-1)[m].astype(np.float64)
            ok = np.isfinite(t) & np.isfinite(q)   # 값이 발산하면 여기서 걸러진다
            if ok.sum() < 8: return float("nan")
            t, q = t[ok], q[ok]
            vt = float(np.var(t))
            if vt < 1e-6: return float("nan")      # 판이 전부 같은 결과 = 잴 게 없다
            return float(np.clip(1.0 - np.var(t - q) / vt, -9.99, 1.0))
        ev0, ev1 = _ev(bv0, rt0, ma0), _ev(bv1, rt1, ma1)
        ppo(net0, opt0, bo0, ba0, bl0, ad0, rt0, ma0, args, dev, pol_on)
        if args.p1_script == "none":
            ppo(net1, opt1, bo1, ba1, bl1, ad1, rt1, ma1, args, dev, pol_on)

        # ── 통계 ──
        # 판 하나가 롤아웃보다 길면 이번 창에서 끝난 판이 하나도 없을 수 있습니다.
        # 그래도 이동량·행동비율은 매 업데이트마다 나와야 하므로, 판 관련 값은
        # '최근 끝난 판' 창(계약서의 '최근 구간')으로 계산해서 한 줄씩 꼭 남깁니다.
        for e in envs:
            recent.extend(e.finished)
            ep_done += len(e.finished)
            e.finished = []
        fin = list(recent)
        el = time.time() - t_start
        p0h = h0 / max(1, h0.sum()); p1h = h1 / max(1, h1.sum())
        if fin:
            n = len(fin)
            row = {"t": "upd", "upd": upd, "ep": ep_done,
                   "ret0": round(float(np.mean([f[0] for f in fin])), 3),
                   "ret1": round(float(np.mean([f[1] for f in fin])), 3),
                   "wr0": round(sum(1 for f in fin if f[2] == 1) / n, 4),
                   "to": round(sum(1 for f in fin if f[2] == 3) / n, 4),
                   "len": round(float(np.mean([f[7] for f in fin])), 2),
                   "ent0": round(ent0_sum / max(1, ent_n), 4),
                   "ent1": round(ent1_sum / max(1, ent_n), 4),
                   "acts0": [round(float(x), 4) for x in p0h],
                   "acts1": [round(float(x), 4) for x in p1h],
                   "mf0": round(float(np.mean([f[5] for f in fin])), 4),
                   "mf1": round(float(np.mean([f[6] for f in fin])), 4),
                   "move0": round(mv0 / max(1, mvn0), 2),
                   "move1": round(mv1 / max(1, mvn1), 2),
                   "ev0": None if ev0 != ev0 else round(ev0, 3),
                   "ev1": None if ev1 != ev1 else round(ev1, 3),
                   "el": round(el, 1), "eps": round(ep_done / max(1e-6, el), 2)}
            emit(row)
            evtxt = "  -- " if row["ev0"] is None else f"{row['ev0']:+.2f}"
            print(f"[{upd:3d}] 판 {ep_done:6d} | 점수 P0 {row['ret0']:+7.2f} P1 {row['ret1']:+7.2f} | "
                  f"P0승률 {row['wr0']*100:5.1f}% 시간초과 {row['to']*100:4.1f}% | "
                  f"이동 {row['move0']:5.1f}/{row['move1']:5.1f}px | "
                  f"탐험 {row['ent0']:.2f}/{row['ent1']:.2f} | "
                  f"값EV {evtxt} | "
                  f"{row['eps']:.1f} 판/초")

            # ── 커리큘럼 단계 전환 ──────────────────────────────────
            # ★ 승률 조건만 걸면 안 된다. 지난번 마린 편에서 28판 만에 통과해서
            #   아무것도 못 배운 채 어려운 단계로 올라갔다. 최소 판수를 같이 건다.
            promote = False
            min_ok = bool(stages) and ep_done - stage_start_ep >= stages[stage][2]
            if stages and args.stage_eval > 0:
                # 최소 판수를 채운 뒤, N 업데이트마다 굳힘으로 재서 승격을 판정한다.
                # 마지막 단계에서도 계속 재서 믿을 수 있는 진행 곡선을 남긴다.
                # ★ 되돌림 가드가 켜져 있으면 최소 판수 전에도 잰다 — 승격 직후 160판 만에
                #   100%→10% 로 무너진 적이 있다 (투혼 마린 14→15). 이 단계 최고 세대를 들고
                #   있다가 그보다 stage_guard 만큼 떨어지면 그 세대로 되돌리고 옵티마이저를 새로 만든다.
                if (min_ok or args.stage_guard > 0) and upd % args.stage_eval_every == 0:
                    gw, gn = greedy_eval(net0, extra_for(_cur["mines"]), args.stage_eval,
                                         seed_off=upd * 4)
                    emit({"t": "geval", "ep": ep_done, "stage": _cur["mines"],
                          "wr": round(gw, 3), "n": gn})
                    print(f"   [굳힘 평가] 단계 {_cur['mines']} · {gn}판 · 승률 {gw*100:.0f}% "
                          f"(학습로그 {row['wr0']*100:.0f}%)", flush=True)
                    if args.stage_guard > 0:
                        if gw > guard["best"] or guard["sd"] is None:
                            # ★ 한 번 튄 점수로 최고를 갈아치우지 않는다. HR1 은 80판 한 번 54% 가 나온 상태를
                            #   붙잡고, 실제로 62% 인 세대(v182)까지 15번 버렸다. 씨앗을 바꿔 한 번 더 재서 합친다.
                            gw2, gn2 = greedy_eval(net0, extra_for(_cur["mines"]), args.stage_eval,
                                                   seed_off=upd * 4 + 3)
                            gwa = (gw * gn + gw2 * gn2) / max(1, gn + gn2)
                            print(f"   [굳힘 재확인·최고 후보] {gn + gn2}판 · 승률 {gwa*100:.0f}%", flush=True)
                            if gwa > guard["best"] or guard["sd"] is None:
                                guard_set(gwa, gn + gn2, _cur["mines"], "평균 갱신")
                            gw, gn = gwa, gn + gn2
                        elif guard["best"] - gw >= args.stage_guard:
                            # ★ 20판 평가는 ±20%p 흔들린다 — 한 번 더 재서 합쳐도 떨어져 있을 때만 되돌린다.
                            #   (FS_kite3: 노이즈로 1,700판에 5번 되돌려 학습 진도를 계속 버렸다)
                            # 씨앗을 바꿔야 한다 — 같은 씨앗이면 같은 판을 또 돌아 같은 숫자가 나온다 (FS_kite3b 실측)
                            gw2, gn2 = greedy_eval(net0, extra_for(_cur["mines"]), args.stage_eval,
                                                                   seed_off=upd * 4 + 1)
                            gw = (gw * gn + gw2 * gn2) / max(1, gn + gn2); gn += gn2
                            print(f"   [굳힘 재확인] {gn}판 · 승률 {gw*100:.0f}%", flush=True)
                            if guard["best"] - gw >= args.stage_guard:
                                net0.load_state_dict(guard["sd"])
                                opt0 = torch.optim.Adam(net0.parameters(), lr=args.lr, eps=1e-5)
                                guard["n"] += 1
                                print(f"   ★ 되돌림 {guard['n']}회: 굳힘 {gw*100:.0f}% → 최고 {guard['best']*100:.0f}% 세대로", flush=True)
                                emit({"t": "guard", "ep": ep_done, "stage": _cur["mines"],
                                      "wr": round(gw, 3), "best": round(guard["best"], 3)})
                    promote = min_ok and (stage + 1 < len(stages)) and gw >= stages[stage][1]
                    if promote and gn <= args.stage_eval:          # 아직 한 번만 잰 점수면 한 번 더
                        gw2, gn2 = greedy_eval(net0, extra_for(_cur["mines"]), args.stage_eval,
                                               seed_off=upd * 4 + 5)
                        gwa = (gw * gn + gw2 * gn2) / max(1, gn + gn2)
                        print(f"   [승격 재확인] {gn + gn2}판 · 승률 {gwa*100:.0f}%", flush=True)
                        promote = gwa >= stages[stage][1]
                    if promote and args.promote_next_min > 0 and stage + 1 < len(stages):
                        nm = stages[stage + 1][0]
                        nw, nn = greedy_eval(net0, extra_for(nm), args.stage_eval, seed_off=upd * 4 + 7)
                        print(f"   [다음 계단 미리보기] {args.curriculum_key}={nm} · {nn}판 · 승률 {nw*100:.0f}% "
                              f"(기준 {args.promote_next_min*100:.0f}%)", flush=True)
                        promote = nw >= args.promote_next_min
            elif min_ok:
                promote = (stage + 1 < len(stages)) and row["wr0"] >= stages[stage][1]
            if promote:
                if args.reset_value_on_stage:
                    net0.reset_value()
                stage += 1
                cur_mines = stages[stage][0]; _cur["mines"] = cur_mines
                stage_start_ep = ep_done
                print(f"   ★ 단계 전환 → {args.curriculum_key}={cur_mines} "
                      f"(승률 {row['wr0']*100:.0f}%, 누적 {ep_done:,}판)", flush=True)
                emit({"t": "stage", "ep": ep_done, "mines": cur_mines,
                      "wr": round(row["wr0"], 3)})
                save_gen(f"단계 전환 마인{cur_mines}개", p0h, p1h)
                for e in envs: e.close()
                envs = make_envs()
                for i, e in enumerate(envs):
                    a, b, _, c, d, _, _ = e.poll()
                    obs0[i], alv0[i], obs1[i], alv1[i] = a, b, c, d
                if args.stage_guard > 0 and args.stage_eval > 0:
                    # 새 단계 진입 기준: 승격 세대가 새 단계에서 얼마나 하는지를 최고점으로 잡아 둔다
                    gw, gn = greedy_eval(net0, extra_for(cur_mines), args.stage_eval,
                                         seed_off=upd * 4 + 2)
                    guard_set(gw, gn, cur_mines, "단계 진입"); guard["n"] = 0
                    emit({"t": "geval", "ep": ep_done, "stage": cur_mines, "wr": round(gw, 3), "n": gn})
                    print(f"   [굳힘 평가] 단계 {cur_mines} 진입 기준 · {gn}판 · 승률 {gw*100:.0f}%", flush=True)

        # ── 세대 저장: 행동이 실제로 달라진 지점에서만 ──
        # ★ 한 업데이트의 행동 분포는 표본이 적어서 그냥 두면 L1 이 흔들리기만 해도 튑니다.
        #   (환경 2개면 벌처 표본이 256개뿐이라 L1 이 저절로 0.2 씩 움직입니다.)
        #   그래서 몇 업데이트를 지수평활한 값으로 비교합니다. 안 그러면 똑같은 세대가 쏟아집니다.
        sm0 = p0h if sm0 is None else 0.7 * sm0 + 0.3 * p0h
        sm1 = p1h if sm1 is None else 0.7 * sm1 + 0.3 * p1h
        if hist0 is None:
            hist0, hist1 = sm0, sm1
        else:
            l0 = float(np.abs(sm0 - hist0).sum()); l1 = float(np.abs(sm1 - hist1).sum())
            if max(l0, l1) >= args.behavior_thresh:
                save_gen(f"행동변화 L1={max(l0, l1):.2f}", sm0, sm1)
                hist0, hist1 = sm0, sm1

    # ── 마무리 ──
    h0 = sm0 if sm0 is not None else np.ones(N_ACT) / N_ACT
    h1 = sm1 if sm1 is not None else np.ones(N_ACT) / N_ACT
    save_gen("종료", h0, h1)
    emit({"t": "stop", "ep": ep_done, "why": "사용자" if stop["flag"] else "예정된 횟수"})
    for e in envs:
        e.close()
    print(f"끝났습니다. 판 {ep_done:,} · {time.time()-t_start:.0f}초 · 세대 {gen[0]}개 · {out}")
    return 0


if __name__ == "__main__":
    _check_files()   # 게임 파일·빌드 확인을 먼저
    sys.exit(main())
