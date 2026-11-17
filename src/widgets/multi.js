// ===== Interactive 5: snack picker (trains live on page load) =====
(function() {
  const dataset = [
    {x:[0,0,0], y:0}, {x:[1,0,0], y:0},
    {x:[0,1,0], y:1}, {x:[1,1,0], y:1},
    {x:[0,0,1], y:2}, {x:[1,0,1], y:2},
    {x:[0,1,1], y:2}, {x:[1,1,1], y:2},
  ];
  const H = 6, C = 3;
  let W1 = [[],[],[]];
  for (let i=0;i<3;i++) for (let j=0;j<H;j++) W1[i].push(Math.random()*2-1);
  let b1 = new Array(H).fill(0);
  let W2 = [];
  for (let j=0;j<H;j++) {
    const row = [];
    for (let k=0;k<C;k++) row.push(Math.random()*2-1);
    W2.push(row);
  }
  let b2 = new Array(C).fill(0);

  function softmax(z) {
    const m = Math.max(...z);
    const e = z.map(v => Math.exp(v - m));
    const s = e.reduce((a,b) => a+b, 0);
    return e.map(v => v/s);
  }

  function forward(x) {
    const h = [];
    for (let j=0;j<H;j++) {
      let z = b1[j];
      for (let i=0;i<3;i++) z += x[i]*W1[i][j];
      h.push(sigmoid(z));
    }
    const z2 = [];
    for (let k=0;k<C;k++) {
      let z = b2[k];
      for (let j=0;j<H;j++) z += h[j]*W2[j][k];
      z2.push(z);
    }
    return { h, p: softmax(z2) };
  }

  function train(epochs) {
    const lr = 0.5;
    for (let ep=0; ep<epochs; ep++) {
      const dW1 = W1.map(r => r.map(()=>0));
      const db1 = b1.map(()=>0);
      const dW2 = W2.map(r => r.map(()=>0));
      const db2 = b2.map(()=>0);
      for (const {x,y} of dataset) {
        const {h, p} = forward(x);
        const target = [0,0,0]; target[y] = 1;
        const d_o = p.map((v,k) => (v - target[k]) / dataset.length);
        for (let k=0;k<C;k++) {
          for (let j=0;j<H;j++) dW2[j][k] += d_o[k] * h[j];
          db2[k] += d_o[k];
        }
        for (let j=0;j<H;j++) {
          let sum = 0;
          for (let k=0;k<C;k++) sum += d_o[k]*W2[j][k];
          const d_h = sum * h[j] * (1 - h[j]);
          for (let i=0;i<3;i++) dW1[i][j] += d_h * x[i];
          db1[j] += d_h;
        }
      }
      for (let j=0;j<H;j++) {
        for (let k=0;k<C;k++) W2[j][k] -= lr*dW2[j][k];
        b1[j] -= lr*db1[j];
        for (let i=0;i<3;i++) W1[i][j] -= lr*dW1[i][j];
      }
      for (let k=0;k<C;k++) b2[k] -= lr*db2[k];
    }
  }
  train(5000);

  const state = { hungry: 0, school: 0, weekend: 0 };
  function bindToggle(id, key) {
    const el = document.getElementById(id);
    el.addEventListener('click', () => {
      state[key] = 1 - state[key];
      el.classList.toggle('on');
      update();
    });
  }
  bindToggle('s-hungry', 'hungry');
  bindToggle('s-school', 'school');
  bindToggle('s-weekend', 'weekend');

  function update() {
    const { p } = forward([state.hungry, state.school, state.weekend]);
    const names = ['fruit', 'chips', 'cookie'];
    const ids = ['s-fruit', 's-chips', 's-cookie'];
    for (let i=0;i<3;i++) {
      document.getElementById(ids[i]+'-bar').style.width = (p[i]*100)+'%';
      document.getElementById(ids[i]+'-pct').textContent = p[i].toFixed(2);
    }
    const best = p.indexOf(Math.max(...p));
    document.getElementById('s-pick').textContent = names[best];
  }
  update();
})();

