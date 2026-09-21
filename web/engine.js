// 전투 판 엔진. 파이썬 aiarena/core/world.py 를 그대로 옮긴 것이다.
// 옮긴 게 맞는지는 대전표로 채점한다 (verify.html 참고).
//
// 라이브러리 없음. 서버 없음. 전부 이 파일 안에서 돈다.

export const ORD_STOP = 0;    // 명령 없음 — 스스로 적을 찾아 쫓아가 싸운다
export const ORD_MOVE = 1;    // 목표로 간다. 가는 동안은 쏘지 않는다
export const ORD_ATTACK = 2;  // 지정한 적을 쏜다. 사거리 밖이면 아무것도 안 한다

export const RESULT_RUNNING = 0;
export const RESULT_P0_WIN = 1;
export const RESULT_P1_WIN = 2;
export const RESULT_TIMEOUT = 3;

// 이 속력보다 빠르게 움직이는 동안은 공격이 나가지 않는다.
// 이게 무빙샷의 대가다 — 한 방 쏘려면 서는 시간과 다시 붙이는 시간을 낸다.
export const FIRE_SPEED_MAX = 2.0;

export const MOVE_PX = 96;
export const N_ACTIONS = 12;
export const DIRS = [
  [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1],
];
export const ACT_NAMES = [
  '정지', '이동1', '이동2', '이동3', '이동4', '이동5', '이동6', '이동7', '이동8',
  '최근접공격', '최저HP공격', '후퇴',
];

// 유닛 도감 — 숫자는 원본 게임 데이터에서 읽은 값이다
export const UNITS = {
  raider: { name: 'raider', hp: 80, damage: 20, cooldown: 30, range: 160,
            speed: 6.668, size: 32, sight: 256, acquire: 160, accel: 0.3906, turn: 56.25 },
  biter:  { name: 'biter',  hp: 35, damage: 5,  cooldown: 8,  range: 15,
            speed: 5.49,  size: 16, sight: 160, acquire: 128, accel: 0,      turn: 37.97 },
};

export const DEFAULT_SCENARIO = {
  key: 'raider_vs_biter',
  p0Unit: 'raider', p0Count: 1,
  p1Unit: 'biter',  p1Count: 6,
  gapLo: 220, gapHi: 300, spreadLo: 34, spreadHi: 48,
  maxSteps: 200, frameSkip: 4,
  cx: 1024, cy: 1024, mapW: 2048, mapH: 2048,
};

// ── 작은 난수 (스폰 값만 정한다) ─────────────────────────────────────
export function makeRng(seed) {
  // xorshift32. 시드를 한 번 섞고 몇 번 버린 뒤 쓴다 —
  // 그냥 쓰면 이웃한 시드끼리 첫 값이 비슷하게 나와서 스폰이 한쪽으로 쏠린다.
  let s = (seed >>> 0) || 1;
  s = (s ^ 0x9e3779b9) >>> 0;
  const next = () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;              // ★ 부호 없는 시프트여야 한다
    s ^= s << 5;  s >>>= 0;
    return s / 4294967296;
  };
  next(); next(); next();
  return next;
}

function half(u) { return u.spec.size / 2; }

export function boxDistance(a, b) {
  let dx = Math.abs(a.x - b.x) - (half(a) + half(b));
  let dy = Math.abs(a.y - b.y) - (half(a) + half(b));
  if (dx < 0) dx = 0;
  if (dy < 0) dy = 0;
  return Math.hypot(dx, dy);
}

export function centerDistance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function dirAction(dx, dy) {
  const n = Math.hypot(dx, dy);
  if (n < 1e-6) return 0;
  dx /= n; dy /= n;
  let best = -9, bestI = 3;
  for (let i = 0; i < DIRS.length; i++) {
    const [ax, ay] = DIRS[i];
    const an = Math.hypot(ax, ay);
    const d = (dx * ax + dy * ay) / an;
    if (d > best) { best = d; bestI = i + 1; }
  }
  return bestI;
}

export class World {
  constructor(scenario, units) {
    this.sc = Object.assign({}, DEFAULT_SCENARIO, scenario || {});
    this.units = units || UNITS;
    this.teams = [[], []];
    this.stepCount = 0;
    this.frame = 0;
    this.result = RESULT_RUNNING;
    this.misfire = [0, 0];
  }

  reset(seed, gap, spread) {
    const sc = this.sc;
    const rnd = makeRng(seed);
    if (gap == null) gap = sc.gapLo + Math.floor(rnd() * (sc.gapHi - sc.gapLo + 1));
    if (spread == null) spread = sc.spreadLo + Math.floor(rnd() * (sc.spreadHi - sc.spreadLo + 1));
    this.gap = gap; this.spread = spread;

    this.teams = [[], []];
    const sides = [
      [sc.p0Unit, sc.p0Count, -1],
      [sc.p1Unit, sc.p1Count, +1],
    ];
    for (let team = 0; team < 2; team++) {
      const [key, count, sign] = sides[team];
      const spec = this.units[key];
      const y0 = sc.cy - ((count - 1) * spread) / 2;
      for (let i = 0; i < count; i++) {
        this.teams[team].push({
          idx: i, team, spec,
          x: sc.cx + sign * Math.floor(gap / 2),
          y: y0 + i * spread,
          hp: spec.hp, cd: 0, alive: true,
          order: ORD_STOP, tx: 0, ty: 0,
          target: null, attacker: null,
          speedNow: 0, heading: 0,
        });
      }
    }
    this.stepCount = 0;
    this.frame = 0;
    this.result = RESULT_RUNNING;
    this.misfire = [0, 0];
  }

  aliveUnits(team) { return this.teams[team].filter((u) => u.alive); }

  nearestFoe(u) {
    let best = null, bd = Infinity;
    for (const z of this.teams[1 - u.team]) {
      if (!z.alive) continue;
      const d = centerDistance(u, z);
      if (d < bd) { bd = d; best = z; }
    }
    return best;
  }

  weakestFoe(u) {
    let best = null, bh = Infinity;
    for (const z of this.teams[1 - u.team]) {
      if (!z.alive) continue;
      if (z.hp < bh) { bh = z.hp; best = z; }
    }
    return best;
  }

  inRange(u, t) {
    if (!t || !t.alive) return false;
    return boxDistance(u, t) <= u.spec.range;
  }

  applyActions(team, actions, strictFire = true) {
    const units = this.teams[team];
    for (let i = 0; i < units.length; i++) {
      const u = units[i];
      if (!u.alive) continue;
      let a = ((actions[i] | 0) % N_ACTIONS + N_ACTIONS) % N_ACTIONS;

      if (a === 0) { u.order = ORD_STOP; u.target = null; continue; }

      if (a >= 1 && a <= 8) {
        const [dx, dy] = DIRS[a - 1];
        u.order = ORD_MOVE;
        u.tx = u.x + dx * MOVE_PX;
        u.ty = u.y + dy * MOVE_PX;
        u.target = null;
        continue;
      }

      if (a === 11) {
        const z = this.nearestFoe(u);
        if (!z) { u.order = ORD_STOP; u.target = null; continue; }
        const dx = u.x - z.x, dy = u.y - z.y;
        const L = Math.max(1, Math.hypot(dx, dy));
        u.order = ORD_MOVE;
        u.tx = u.x + (dx / L) * MOVE_PX;
        u.ty = u.y + (dy / L) * MOVE_PX;
        u.target = null;
        continue;
      }

      const z = a === 9 ? this.nearestFoe(u) : this.weakestFoe(u);
      if (strictFire && !this.inRange(u, z)) {
        // 사거리 밖에서 쏘라고 했다 = 헛방. 제자리에 선다.
        this.misfire[team]++;
        u.order = ORD_STOP; u.target = null;
        continue;
      }
      u.order = ORD_ATTACK; u.target = z;
    }
  }

  tick() {
    const order = [];
    for (const t of this.teams) for (const u of t) if (u.alive) order.push(u);

    for (const u of order) {
      if (u.order === ORD_MOVE) this._steer(u, u.tx, u.ty);
      else if (u.order === ORD_ATTACK) this._brake(u);
    }

    for (const u of order) {
      if (!u.alive) continue;
      if (u.order === ORD_ATTACK) this._tryFire(u, u.target);
      else if (u.order === ORD_STOP) this._stopBehavior(u);
    }

    for (const u of order) if (u.cd > 0) u.cd--;

    this._separate();
    this._clamp();
    this.frame++;
  }

  // 명령이 없는 유닛이 스스로 하는 일.
  // 원본 게임의 '정지'는 제자리에 못 박혀 있는 게 아니다 — 적을 쫓아가고, 맞으면 반격하러 간다.
  _stopBehavior(u) {
    const t = this._autoTarget(u);
    if (t) { this._brake(u); this._tryFire(u, t); return; }

    const chase = u.spec.acquire * 2;
    let z = null;
    if (u.attacker && u.attacker.alive && boxDistance(u, u.attacker) <= chase) {
      z = u.attacker;
    } else {
      u.attacker = null;
      let best = null, bd = Infinity;
      for (const f of this.teams[1 - u.team]) {
        if (!f.alive) continue;
        const d = boxDistance(u, f);
        if (d <= u.spec.acquire && d < bd) { bd = d; best = f; }
      }
      z = best;
    }
    if (!z) { this._brake(u); return; }
    this._steer(u, z.x, z.y);
  }

  // 걸어다니는 유닛(accel=0)은 즉시 방향을 바꾼다.
  // 미끄러지는 유닛은 멈췄다 가는 게 아니라 **선회**한다. 이게 무빙샷이 성립하는 이유다.
  _steer(u, tx, ty) {
    const spd = u.spec.speed, acc = u.spec.accel;
    const dx = tx - u.x, dy = ty - u.y;
    const d = Math.hypot(dx, dy);

    if (acc <= 0) {
      if (d <= 1e-6) { u.speedNow = 0; return; }
      u.heading = Math.atan2(dy, dx);
      const step = d > spd ? spd : d;
      u.speedNow = step;
      u.x += Math.cos(u.heading) * step;
      u.y += Math.sin(u.heading) * step;
      return;
    }

    if (d <= 1e-6) { this._brake(u); return; }

    const want = Math.atan2(dy, dx);
    const turn = (u.spec.turn * Math.PI) / 180;
    let diff = ((want - u.heading + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    if (diff > turn) diff = turn;
    else if (diff < -turn) diff = -turn;
    u.heading += diff;

    u.speedNow = Math.min(spd, u.speedNow + acc);
    const step = u.speedNow;
    if (step >= d) { u.x = tx; u.y = ty; u.speedNow = 0; return; }
    u.x += Math.cos(u.heading) * step;
    u.y += Math.sin(u.heading) * step;
  }

  _brake(u) {
    const acc = u.spec.accel;
    if (acc <= 0 || u.speedNow <= acc) { u.speedNow = 0; return; }
    u.speedNow -= acc;
    u.x += Math.cos(u.heading) * u.speedNow;
    u.y += Math.sin(u.heading) * u.speedNow;
  }

  _autoTarget(u) {
    let best = null, bd = Infinity;
    for (const z of this.teams[1 - u.team]) {
      if (!z.alive) continue;
      const d = boxDistance(u, z);
      if (d <= u.spec.range && d < bd) { bd = d; best = z; }
    }
    return best;
  }

  _tryFire(u, t) {
    if (!t || !t.alive || u.cd > 0) return;
    if (u.speedNow > FIRE_SPEED_MAX) return;   // 달리면서는 못 쏜다
    if (boxDistance(u, t) > u.spec.range) return;
    t.hp -= u.spec.damage;
    t.attacker = u;
    u.cd = u.spec.cooldown;
    if (t.hp <= 0) { t.alive = false; t.hp = 0; }
  }

  _separate() {
    const units = [];
    for (const t of this.teams) for (const u of t) if (u.alive) units.push(u);
    for (let i = 0; i < units.length; i++) {
      const a = units[i];
      for (let j = i + 1; j < units.length; j++) {
        const b = units[j];
        const rx = half(a) + half(b);
        const dx = b.x - a.x, dy = b.y - a.y;
        const ox = rx - Math.abs(dx);
        const oy = rx - Math.abs(dy);
        if (ox <= 0 || oy <= 0) continue;
        if (ox < oy) {
          const push = ox / 2, s = dx >= 0 ? 1 : -1;
          a.x -= push * s; b.x += push * s;
        } else {
          const push = oy / 2, s = dy >= 0 ? 1 : -1;
          a.y -= push * s; b.y += push * s;
        }
      }
    }
  }

  _clamp() {
    const w = this.sc.mapW, h = this.sc.mapH;
    for (const t of this.teams) for (const u of t) {
      if (!u.alive) continue;
      const hf = half(u);
      if (u.x < hf) u.x = hf; else if (u.x > w - hf) u.x = w - hf;
      if (u.y < hf) u.y = hf; else if (u.y > h - hf) u.y = h - hf;
    }
  }

  advance() {
    for (let i = 0; i < this.sc.frameSkip; i++) {
      this.tick();
      const r = this._checkEnd();
      if (r !== RESULT_RUNNING) { this.result = r; return r; }
    }
    this.stepCount++;
    if (this.stepCount >= this.sc.maxSteps) {
      this.result = RESULT_TIMEOUT;
      return RESULT_TIMEOUT;
    }
    return RESULT_RUNNING;
  }

  _checkEnd() {
    const a0 = this.teams[0].some((u) => u.alive);
    const a1 = this.teams[1].some((u) => u.alive);
    if (!a1 && !a0) return RESULT_TIMEOUT;
    if (!a1) return RESULT_P0_WIN;
    if (!a0) return RESULT_P1_WIN;
    return RESULT_RUNNING;
  }
}

// ── 관측 조립 (파이썬 core/obs.py 와 같은 39칸) ──────────────────────
export const OBS_DIM = 39;
const SELF_DIM = 5, FOE_DIM = 6, ALLY_DIM = 4, N_FOE = 3, N_ALLY = 4;

export function buildObs(w, team) {
  const units = w.teams[team];
  const meSpec = units[0].spec;
  const foes = w.teams[1 - team];
  const foeMaxHp = foes.length ? foes[0].spec.hp : 1;
  const cdMax = Math.max(1, meSpec.cooldown);
  const { cx, cy } = w.sc;

  const out = [];
  for (const u of units) {
    // ★ float32 다. 파이썬 쪽 관측이 float32 라 여기서 자리수를 맞춰야
    //   같은 판에서 같은 판단이 나온다. float64 로 두면 8방향 선택이
    //   경계에서 갈려 결과가 몇 판씩 달라진다.
    const o = new Float32Array(OBS_DIM);
    if (!u.alive) { out.push(o); continue; }
    o[0] = u.hp / meSpec.hp;
    o[1] = Math.min(u.cd, cdMax) / cdMax;
    o[2] = 1;
    o[3] = (u.x - cx) / 512;
    o[4] = (u.y - cy) / 512;

    const zs = foes.filter((z) => z.alive)
      .map((z) => [centerDistance(u, z), z])
      .sort((a, b) => a[0] - b[0]);

    for (let k = 0; k < Math.min(N_FOE, zs.length); k++) {
      const [d, z] = zs[k];
      const q = SELF_DIM + k * FOE_DIM;
      o[q + 0] = (z.x - u.x) / 256;
      o[q + 1] = (z.y - u.y) / 256;
      o[q + 2] = d / 256;
      o[q + 3] = z.hp / foeMaxHp;
      o[q + 4] = 1;
      o[q + 5] = boxDistance(u, z) <= meSpec.range ? 1 : 0;
    }

    let c = 0;
    const base = SELF_DIM + N_FOE * FOE_DIM;
    for (const m of units) {
      if (m === u || !m.alive || c >= N_ALLY) continue;
      const q = base + c * ALLY_DIM;
      o[q + 0] = (m.x - u.x) / 256;
      o[q + 1] = (m.y - u.y) / 256;
      o[q + 2] = m.hp / meSpec.hp;
      o[q + 3] = 1;
      c++;
    }
    out.push(o);
  }
  return out;
}

export function aliveMask(w, team) {
  return w.teams[team].map((u) => (u.alive ? 1 : 0));
}
