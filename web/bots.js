// 기준선 봇 4종. 파이썬 aiarena/bots/baselines.py 와 같다.
// 넷 다 관측만 보고 판단한다 — 엔진 내부를 들여다보지 않는다.

import { dirAction } from './engine.js';

const FOE_DX = 5, FOE_DY = 6, FOE_IN_RANGE = 10, MY_CD = 1;
const ATTACK_NEAR = 9, RETREAT = 11;

// 가만히 선다. 사거리에 적이 들어오면 알아서 쏘고, 안 들어오면 쫓아간다.
export function stop(obs) {
  return obs.map(() => 0);
}

// 사거리에 들어오면 쏘고, 아니면 가장 가까운 적에게 직진한다.
export function rush(obs, mask) {
  return obs.map((row, i) => {
    if (!mask[i]) return 0;
    return row[FOE_IN_RANGE] > 0.5
      ? ATTACK_NEAR
      : dirAction(row[FOE_DX] * 256, row[FOE_DY] * 256);
  });
}

// 무빙샷. 쏠 수 있으면 쏘고, 쿨다운 도는 동안은 물러난다.
export function kite(obs, mask) {
  return obs.map((row, i) => {
    if (!mask[i]) return 0;
    const inRange = row[FOE_IN_RANGE] > 0.5;
    if (inRange && row[MY_CD] < 0.1) return ATTACK_NEAR;
    if (inRange) return RETREAT;
    return dirAction(row[FOE_DX] * 256, row[FOE_DY] * 256);
  });
}

// 돌진하되 홀짝으로 위아래를 벌려서 접근한다.
export function spread(obs, mask) {
  return obs.map((row, i) => {
    if (!mask[i]) return 0;
    if (row[FOE_IN_RANGE] > 0.5) return ATTACK_NEAR;
    return dirAction(row[FOE_DX] * 256, row[FOE_DY] * 256 + (i % 2 ? 60 : -60));
  });
}

export const BASELINES = { stop, rush, kite, spread };
export const BASELINE_KR = { stop: '정지', rush: '돌진', kite: '무빙샷', spread: '벌리기' };
