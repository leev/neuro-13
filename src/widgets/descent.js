// ===== Interactive 6: gradient descent =====
(function() {
  const canvas = document.getElementById('gd-canvas');
  const { ctx, w: W, h: H } = Book.hidpi(canvas);
  const lr = 0.15;
  let x = -3.5;

  function f(x)  { return 0.35*x*x + 0.3; }
  function df(x) { return 0.7*x; }

  function draw() {
    ctx.clearRect(0,0,W,H);
    // grid baseline
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, H-30); ctx.lineTo(W, H-30);
    ctx.stroke();
    // curve
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 3;
    ctx.beginPath();
    let first = true;
    for (let px = 0; px < W; px++) {
      const xv = (px / W) * 9 - 4.5;
      const yv = f(xv);
      const py = H - 30 - (yv * 35);
      if (first) { ctx.moveTo(px, py); first = false; }
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    // dot
    const dotPx = ((x + 4.5) / 9) * W;
    const dotPy = H - 30 - (f(x) * 35);
    ctx.fillStyle = '#dc2626';
    ctx.beginPath();
    ctx.arc(dotPx, dotPy, 11, 0, Math.PI*2);
    ctx.fill();
    ctx.strokeStyle = '#7f1d1d';
    ctx.lineWidth = 2;
    ctx.stroke();
    // labels
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px ui-monospace, Menlo, monospace';
    ctx.fillText('weight = '+x.toFixed(2), 16, 26);
    ctx.fillText('error  = '+f(x).toFixed(2), 16, 46);
    ctx.font = '11px sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.fillText('← lower weight', 10, H-10);
    ctx.fillText('higher weight →', W-100, H-10);
  }

  function step(n=1) {
    for (let i=0;i<n;i++) x -= lr * df(x);
    draw();
  }
  function reset() { x = -3.5; draw(); }
  document.getElementById('gd-step').addEventListener('click', () => step(1));
  document.getElementById('gd-fast').addEventListener('click', () => step(20));
  document.getElementById('gd-reset').addEventListener('click', reset);
  draw();
})();

