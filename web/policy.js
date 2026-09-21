// 학습한 망을 브라우저에서 돌린다.
//
// 망이 39 → 128 → 128 → 12 밖에 안 돼서 라이브러리가 필요 없다.
// 행렬곱 세 번이면 끝이고, 한 번 추론에 1ms 도 안 걸린다.

export class Net {
  constructor(data) {
    this.shape = data.shape;           // [39, 128, 128, 12]
    this.w0 = Float64Array.from(data.w0); this.b0 = Float64Array.from(data.b0);
    this.w1 = Float64Array.from(data.w1); this.b1 = Float64Array.from(data.b1);
    this.w2 = Float64Array.from(data.w2); this.b2 = Float64Array.from(data.b2);
    this._h0 = new Float64Array(this.shape[1]);
    this._h1 = new Float64Array(this.shape[2]);
    this._out = new Float64Array(this.shape[3]);
  }

  // torch 의 Linear 는 (출력, 입력) 순서로 가중치를 담는다. 그대로 맞춘다.
  _linear(x, w, b, out, nIn, nOut) {
    for (let o = 0; o < nOut; o++) {
      let s = b[o];
      const base = o * nIn;
      for (let i = 0; i < nIn; i++) s += w[base + i] * x[i];
      out[o] = s;
    }
    return out;
  }

  // 관측 한 줄 -> 행동 번호 (확률이 제일 높은 것)
  act(obs) {
    const [n0, n1, n2, n3] = this.shape;
    this._linear(obs, this.w0, this.b0, this._h0, n0, n1);
    for (let i = 0; i < n1; i++) this._h0[i] = Math.tanh(this._h0[i]);
    this._linear(this._h0, this.w1, this.b1, this._h1, n1, n2);
    for (let i = 0; i < n2; i++) this._h1[i] = Math.tanh(this._h1[i]);
    this._linear(this._h1, this.w2, this.b2, this._out, n2, n3);

    let best = 0, bv = this._out[0];
    for (let i = 1; i < n3; i++) if (this._out[i] > bv) { bv = this._out[i]; best = i; }
    return best;
  }
}

// 망을 봇으로 감싼다. 봇 계약은 손코딩과 똑같다.
export function makeBot(net) {
  return (obs, mask) => obs.map((row, i) => (mask[i] ? net.act(row) : 0));
}

export async function loadNet(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} 를 못 읽었습니다 (${r.status})`);
  return new Net(await r.json());
}
