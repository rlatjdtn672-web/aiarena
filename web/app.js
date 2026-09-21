// 화면과 조작. 엔진은 engine.js, 봇은 bots.js, 학습한 망은 policy.js 가 맡는다.
//
// 두 가지가 동시에 돌아간다:
//   1) 한 판을 천천히 그려서 보여준다
//   2) 백그라운드로 24판을 즉시 돌려 승률을 낸다 (판당 1ms 도 안 걸린다)
// 숫자를 밀면 둘 다 새로 시작한다.

import {
  World, buildObs, aliveMask, UNITS, DEFAULT_SCENARIO,
  RESULT_RUNNING, RESULT_P0_WIN, RESULT_P1_WIN,
} from './engine.js';
import { BASELINES } from './bots.js';
import { loadNet, makeBot } from './policy.js';

const $ = (id) => document.getElementById(id);
const cv = $('cv');
const ctx = cv.getContext('2d');

const nets = {};          // 학습한 망 (처음 고를 때 받아온다)
let paused = false;
let speed = 2;
let sim = null;           // 지금 보여주는 판
let acc = 0;
let lastT = 0;

// ── 조작값 ──────────────────────────────────────────────────────
const DEFAULTS = { range: 160, speed: 6.67, count: 6, cooldown: 30, frameskip: 4 };
const knobs = { ...DEFAULTS };

function scenario() {
  return { ...DEFAULT_SCENARIO, p1Count: knobs.count, frameSkip: knobs.frameskip };
}
function units() {
  return {
    raider: { ...UNITS.raider, range: knobs.range, speed: knobs.speed, cooldown: knobs.cooldown },
    biter: { ...UNITS.biter },
  };
}

async function botOf(value) {
  if (value.startsWith('net_')) {
    const key = value.slice(4);
    if (!nets[key]) {
      const file = key === 'ft' ? 'weights/ft.json' : 'weights/clone.json';
      nets[key] = await loadNet(file);
    }
    return makeBot(nets[key]);
  }
  return BASELINES[value];
}

// ── 한 판 (화면용) ───────────────────────────────────────────────
function newSim(bot0, bot1, seed) {
  const w = new World(scenario(), units());
  w.reset(seed);
  return { w, bot0, bot1, done: false };
}

function stepSim(s) {
  if (s.done) return;
  const o0 = buildObs(s.w, 0), m0 = aliveMask(s.w, 0);
  const o1 = buildObs(s.w, 1), m1 = aliveMask(s.w, 1);
  s.w.applyActions(0, s.bot0(o0, m0));
  s.w.applyActions(1, s.bot1(o1, m1));
  if (s.w.advance() !== RESULT_RUNNING) s.done = true;
}

// ── 24판 즉시 집계 ───────────────────────────────────────────────
function runBatch(bot0, bot1, n = 24) {
  const sc = scenario(), un = units();
  let w0 = 0, w1 = 0, to = 0;
  for (let e = 0; e < n; e++) {
    const w = new World(sc, un);
    w.reset(1000 + e * 7919);
    let r = RESULT_RUNNING;
    while (r === RESULT_RUNNING) {
      const o0 = buildObs(w, 0), m0 = aliveMask(w, 0);
      const o1 = buildObs(w, 1), m1 = aliveMask(w, 1);
      w.applyActions(0, bot0(o0, m0));
      w.applyActions(1, bot1(o1, m1));
      r = w.advance();
    }
    if (r === RESULT_P0_WIN) w0++;
    else if (r === RESULT_P1_WIN) w1++;
    else to++;
  }
  return { w0, w1, to, n };
}

// ── 카메라 ──────────────────────────────────────────────────────
// 맵은 2048×2048 인데 싸움은 그 일부에서만 벌어진다. 특히 도망치는 쪽이
// 구석으로 붙으면 전체 맵을 그릴 때 점 하나로 보인다. 그래서 살아 있는
// 유닛을 담는 사각형에 맞춰 확대하고, 튀지 않게 천천히 따라가게 한다.
let cam = null;

function targetCam(w) {
  const us = [...w.teams[0], ...w.teams[1]].filter((u) => u.alive);
  if (!us.length) return cam;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const u of us) {
    x0 = Math.min(x0, u.x); x1 = Math.max(x1, u.x);
    y0 = Math.min(y0, u.y); y1 = Math.max(y1, u.y);
  }
  const pad = 190;
  x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;
  let size = Math.max(x1 - x0, y1 - y0, 560);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  return { cx, cy, size };
}

function updateCam(w, snap) {
  const t = targetCam(w);
  if (!t) return;
  if (!cam || snap) { cam = { ...t }; return; }
  const k = 0.08;
  cam.cx += (t.cx - cam.cx) * k;
  cam.cy += (t.cy - cam.cy) * k;
  cam.size += (t.size - cam.size) * k;
}

// ── 그리기 ──────────────────────────────────────────────────────
function draw(w) {
  const S = cv.width;
  updateCam(w);
  const view = cam || { cx: w.sc.cx, cy: w.sc.cy, size: w.sc.mapW };
  const scale = S / view.size;
  const ox = view.cx - view.size / 2;
  const oy = view.cy - view.size / 2;
  const PX = (x) => (x - ox) * scale;
  const PY = (y) => (y - oy) * scale;
  ctx.clearRect(0, 0, S, S);

  // 바닥 격자 (월드 기준이라 카메라가 움직이면 같이 흐른다 — 속도가 느껴진다)
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  const G = 256;
  for (let gx = Math.floor(ox / G) * G; gx < ox + view.size + G; gx += G) {
    const p = PX(gx);
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, S); ctx.stroke();
  }
  for (let gy = Math.floor(oy / G) * G; gy < oy + view.size + G; gy += G) {
    const p = PY(gy);
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(S, p); ctx.stroke();
  }

  // 맵 경계 — 도망치는 쪽이 여기 막히는 게 승부를 가른다
  ctx.strokeStyle = 'rgba(255,200,97,0.28)';
  ctx.lineWidth = 2;
  ctx.strokeRect(PX(0), PY(0), w.sc.mapW * scale, w.sc.mapH * scale);

  const COLOR = ['#5aa9ff', '#ff6b6b'];
  const GLOW = ['rgba(90,169,255,0.16)', 'rgba(255,107,107,0.14)'];

  // 사거리 원 (파랑만 — 빨강은 거의 붙어야 해서 안 보인다)
  for (const u of w.teams[0]) {
    if (!u.alive) continue;
    ctx.fillStyle = GLOW[0];
    ctx.beginPath();
    ctx.arc(PX(u.x), PY(u.y), (u.spec.range + u.spec.size / 2) * scale, 0, Math.PI * 2);
    ctx.fill();
  }

  for (let t = 0; t < 2; t++) {
    for (const u of w.teams[t]) {
      if (!u.alive) continue;
      const x = PX(u.x), y = PY(u.y);
      const r = Math.max(3, (u.spec.size / 2) * scale);

      // 몸
      ctx.fillStyle = COLOR[t];
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();

      // 바라보는 방향
      if (u.speedNow > 0.2) {
        ctx.strokeStyle = COLOR[t];
        ctx.lineWidth = Math.max(1.5, r * 0.28);
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + Math.cos(u.heading) * r * 2.1, y + Math.sin(u.heading) * r * 2.1);
        ctx.stroke();
      }

      // 체력 막대
      const hpw = r * 2.4, frac = Math.max(0, u.hp / u.spec.hp);
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(x - hpw / 2, y - r - 9, hpw, 4);
      ctx.fillStyle = frac > 0.5 ? '#6ee7a8' : frac > 0.25 ? '#ffc861' : '#ff6b6b';
      ctx.fillRect(x - hpw / 2, y - r - 9, hpw * frac, 4);

      // 쏠 준비가 됐으면 테두리
      if (u.cd === 0 && t === 0) {
        ctx.strokeStyle = '#ffc861';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, r + 3, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }
}

// ── 승률 곡선 ───────────────────────────────────────────────────
// 슬라이더를 밀기 전에 "어디서 뒤집히는지"가 보이게, 값마다 24판씩 돌려 곡선을 그린다.
// 판당 1ms 도 안 걸려서 브라우저에서 바로 계산된다. 화면이 멈추지 않게 조금씩 나눠서 한다.
const SPARK_VALUES = {
  range:     [40, 70, 100, 120, 140, 160, 180, 200, 230, 260],
  speed:     [4.5, 5.0, 5.5, 6.0, 6.3, 6.67, 7.0, 7.5, 8.0, 9.0],
  count:     [2, 3, 4, 5, 6, 7, 8, 10, 12, 14],
  cooldown:  [8, 14, 20, 26, 30, 36, 42, 48, 54, 60],
  frameskip: [1, 2, 3, 4, 5, 6, 8, 10, 12, 16],
};
const sparkData = {};
let sparkQueue = [];
let sparkToken = 0;

function queueSparks() {
  sparkToken++;
  for (const k of Object.keys(SPARK_VALUES)) sparkData[k] = [];
  sparkQueue = [];
  for (const k of Object.keys(SPARK_VALUES)) {
    for (const v of SPARK_VALUES[k]) sparkQueue.push([k, v, sparkToken]);
  }
}

function pumpSparks() {
  if (!sparkQueue.length) return;
  const t0 = performance.now();
  while (sparkQueue.length && performance.now() - t0 < 12) {
    const [k, v, tok] = sparkQueue.shift();
    if (tok !== sparkToken) continue;
    const saved = knobs[k];
    knobs[k] = v;
    const { w0, n } = runBatch(bots.b0, bots.b1, 24);
    knobs[k] = saved;
    sparkData[k].push([v, w0 / n]);
    drawSpark(k);
  }
}

function drawSpark(key) {
  const cvs = document.getElementById('k-' + key);
  if (!cvs) return;
  // 표시되는 크기에 맞춰 해상도를 잡는다 (안 그러면 가로세로가 따로 눌린다)
  const rect = cvs.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const W = Math.max(200, Math.round(rect.width * dpr));
  const H = Math.round((rect.height || 52) * dpr);
  if (cvs.width !== W || cvs.height !== H) { cvs.width = W; cvs.height = H; }
  const c = cvs.getContext('2d');
  c.clearRect(0, 0, W, H);

  const vals = SPARK_VALUES[key];
  const lo = vals[0], hi = vals[vals.length - 1];
  const padX = 10 * dpr, padT = 9 * dpr, padB = 9 * dpr;
  const px = (v) => ((v - lo) / (hi - lo)) * (W - padX * 2) + padX;
  const py = (p) => H - padB - p * (H - padT - padB);

  // 0% · 50% · 100% 기준선
  c.lineWidth = 1 * dpr;
  for (const [p, a] of [[0, 0.09], [0.5, 0.14], [1, 0.09]]) {
    c.strokeStyle = `rgba(255,255,255,${a})`;
    c.setLineDash(p === 0.5 ? [4 * dpr, 4 * dpr] : []);
    c.beginPath(); c.moveTo(0, py(p)); c.lineTo(W, py(p)); c.stroke();
  }
  c.setLineDash([]);

  const pts = sparkData[key] || [];
  if (pts.length > 1) {
    // 면
    c.beginPath();
    c.moveTo(px(pts[0][0]), H);
    for (const [v, p] of pts) c.lineTo(px(v), py(p));
    c.lineTo(px(pts[pts.length - 1][0]), H);
    c.closePath();
    c.fillStyle = 'rgba(90,169,255,0.16)';
    c.fill();
    // 선
    c.beginPath();
    pts.forEach(([v, p], i) => (i ? c.lineTo(px(v), py(p)) : c.moveTo(px(v), py(p))));
    c.strokeStyle = '#5aa9ff';
    c.lineWidth = 2.2 * dpr;
    c.lineJoin = 'round';
    c.stroke();
  }

  // 지금 값
  const cur = knobs[key];
  if (cur >= lo && cur <= hi) {
    c.strokeStyle = '#ffc861';
    c.lineWidth = 1.8 * dpr;
    c.beginPath(); c.moveTo(px(cur), 2); c.lineTo(px(cur), H - 2); c.stroke();
    // 지금 값의 승률을 숫자로
    const near = (sparkData[key] || []).reduce(
      (b, p) => (b === null || Math.abs(p[0] - cur) < Math.abs(b[0] - cur) ? p : b), null);
    if (near) {
      c.fillStyle = '#ffc861';
      c.font = `${11 * dpr}px system-ui`;
      c.textAlign = px(cur) > W * 0.7 ? 'right' : 'left';
      c.fillText(`${Math.round(near[1] * 100)}%`, px(cur) + (px(cur) > W * 0.7 ? -6 * dpr : 6 * dpr), 13 * dpr);
      c.textAlign = 'left';
    }
  }

  if (sparkQueue.length && pts.length < vals.length) {
    c.fillStyle = 'rgba(140,151,171,0.8)';
    c.font = `${11 * dpr}px system-ui`;
    c.fillText('재는 중…', 10 * dpr, 15 * dpr);
  }
}

// ── 루프 ────────────────────────────────────────────────────────
function frame(t) {
  requestAnimationFrame(frame);
  if (!sim) return;
  const dt = Math.min(100, t - lastT);
  lastT = t;
  if (paused) { draw(sim.w); return; }

  acc += dt * speed;
  const per = 42;                 // 한 판단 ≈ 42ms (원본 프레임 간격)
  let guard = 0;
  while (acc >= per && !sim.done && guard++ < 60) {
    stepSim(sim);
    acc -= per;
  }
  draw(sim.w);
  pumpSparks();

  const v = $('verdict');
  if (sim.done) {
    const r = sim.w.result;
    v.textContent = r === RESULT_P0_WIN ? '파랑 승'
      : r === RESULT_P1_WIN ? '빨강 승' : '시간 초과';
    v.className = 'overlay ' + (r === RESULT_P0_WIN ? 'win' : r === RESULT_P1_WIN ? 'lose' : '');
    if (acc > 2200) restartView();         // 결과를 잠깐 보여주고 다시
  } else {
    v.textContent = `${sim.w.aliveUnits(0).length} 대 ${sim.w.aliveUnits(1).length}`;
    v.className = 'overlay';
  }
}

// ── 다시 시작 ───────────────────────────────────────────────────
let viewSeed = 7;
let bots = { b0: BASELINES.kite, b1: BASELINES.spread };

function restartView() {
  viewSeed = (viewSeed * 1103515245 + 12345) & 0x7fffffff;
  sim = newSim(bots.b0, bots.b1, viewSeed);
  updateCam(sim.w, true);
  acc = 0;
}

function refreshBatch() {
  const el = $('wrnum');
  el.textContent = '재는 중…';
  // 그리는 걸 막지 않게 다음 프레임으로 미룬다
  requestAnimationFrame(() => {
    const t0 = performance.now();
    const { w0, w1, to, n } = runBatch(bots.b0, bots.b1);
    const ms = performance.now() - t0;
    const pct = Math.round((w0 / n) * 100);
    $('wrfill').style.width = pct + '%';
    el.innerHTML = `파랑 <b>${w0}</b>승 · 빨강 ${w1}승` +
      (to ? ` · 시간초과 ${to}` : '') +
      ` <span style="color:var(--dim)">(${pct}% · ${ms.toFixed(0)}ms)</span>`;
  });
}

async function reload() {
  bots = {
    b0: await botOf($('p0').value),
    b1: await botOf($('p1').value),
  };
  restartView();
  refreshBatch();
  queueSparks();
  for (const k of Object.keys(SPARK_VALUES)) drawSpark(k);
}

// ── 조작 연결 ───────────────────────────────────────────────────
function bindSlider(key, fmt) {
  const input = $('s-' + key);
  const out = $('v-' + key);
  input.addEventListener('input', () => {
    knobs[key] = parseFloat(input.value);
    out.textContent = fmt(knobs[key]);
    input.closest('.slider').classList.toggle('changed', knobs[key] !== DEFAULTS[key]);
    restartView();
    refreshBatch();
    drawSpark(key);
  });
}

bindSlider('range', (v) => v);
bindSlider('speed', (v) => v.toFixed(2));
bindSlider('count', (v) => v);
bindSlider('cooldown', (v) => v);
bindSlider('frameskip', (v) => v);

$('p0').addEventListener('change', reload);
$('p1').addEventListener('change', reload);
$('replay').addEventListener('click', restartView);
$('pause').addEventListener('click', () => {
  paused = !paused;
  $('pause').textContent = paused ? '재생' : '멈춤';
});
$('speed').addEventListener('input', (e) => {
  speed = parseInt(e.target.value, 10);
  $('speedval').textContent = speed + '배';
});
$('reset').addEventListener('click', () => {
  for (const k of Object.keys(DEFAULTS)) {
    knobs[k] = DEFAULTS[k];
    const input = $('s-' + k);
    input.value = DEFAULTS[k];
    input.dispatchEvent(new Event('input'));
  }
});

// 시작
reload();
requestAnimationFrame((t) => { lastT = t; requestAnimationFrame(frame); });
