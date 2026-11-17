(function neuronSandbox() {
  const ids = ['n-x1','n-x2','n-w1','n-w2','n-b'];
  function update() {
    const x1 = +document.getElementById('n-x1').value;
    const x2 = +document.getElementById('n-x2').value;
    const w1 = +document.getElementById('n-w1').value;
    const w2 = +document.getElementById('n-w2').value;
    const b  = +document.getElementById('n-b').value;
    document.getElementById('n-x1-v').textContent = x1.toFixed(2);
    document.getElementById('n-x2-v').textContent = x2.toFixed(2);
    document.getElementById('n-w1-v').textContent = w1.toFixed(2);
    document.getElementById('n-w2-v').textContent = w2.toFixed(2);
    document.getElementById('n-b-v').textContent  = b.toFixed(2);
    const z = x1*w1 + x2*w2 + b;
    const out = sigmoid(z);
    document.getElementById('n-z').textContent = z.toFixed(2);
    document.getElementById('n-out-bar').style.width = (out*100) + '%';
    document.getElementById('n-out-pct').textContent = out.toFixed(2);
    document.getElementById('n-decision').textContent = out > 0.5 ? 'YES' : 'NO';
    document.getElementById('n-decision').style.color = out > 0.5 ? 'var(--good)' : 'var(--bad)';
    drawNeuronViz(x1, x2, w1, w2, b, z, out);
  }
  function nodeFill(v) {
    const t = Math.max(0, Math.min(1, Math.abs(v)));
    return v >= 0
      ? `rgba(37, 99, 235, ${0.12 + t * 0.78})`
      : `rgba(220, 38, 38, ${0.12 + t * 0.78})`;
  }
  function edgeColor(v) { return v >= 0 ? '#2563eb' : '#dc2626'; }
  function edgeWidth(v) { return Math.min(7, Math.abs(v) * 2.2 + 0.6); }
  function drawNeuronViz(x1, x2, w1, w2, b, z, out) {
    const svg = document.getElementById('n-viz');
    const c1 = x1 * w1;
    const c2 = x2 * w2;
    const decision = out > 0.5 ? 'YES' : 'NO';
    const outFill = out > 0.5 ? '#d1fae5' : '#fee2e2';
    const outStroke = out > 0.5 ? '#16a34a' : '#dc2626';
    svg.innerHTML = `
      <defs>
        <filter id="n-glow" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="4" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <text x="36" y="24" font-size="13" fill="#6b7280">inputs</text>
      <text x="214" y="24" font-size="13" fill="#6b7280">weighted votes</text>
      <text x="372" y="24" font-size="13" fill="#6b7280">add + squish</text>
      <text x="540" y="24" font-size="13" fill="#6b7280">decision</text>

      <line x1="84" y1="78" x2="220" y2="78" stroke="${edgeColor(c1)}" stroke-width="${edgeWidth(c1)}" opacity="0.75" ${Math.abs(c1) > 0.65 ? 'filter="url(#n-glow)"' : ''}/>
      <line x1="84" y1="176" x2="220" y2="176" stroke="${edgeColor(c2)}" stroke-width="${edgeWidth(c2)}" opacity="0.75" ${Math.abs(c2) > 0.65 ? 'filter="url(#n-glow)"' : ''}/>
      <line x1="284" y1="78" x2="360" y2="125" stroke="${edgeColor(c1)}" stroke-width="${edgeWidth(c1)}" opacity="0.65"/>
      <line x1="284" y1="176" x2="360" y2="125" stroke="${edgeColor(c2)}" stroke-width="${edgeWidth(c2)}" opacity="0.65"/>
      <line x1="284" y1="128" x2="360" y2="128" stroke="${edgeColor(b)}" stroke-width="${edgeWidth(b)}" opacity="0.65"/>
      <line x1="414" y1="126" x2="498" y2="126" stroke="#9ca3af" stroke-width="3"/>

      <circle cx="62" cy="78" r="24" fill="${nodeFill(x1)}" stroke="#1f2937" stroke-width="2"/>
      <text x="62" y="46" text-anchor="middle" font-size="12" fill="#1f2937" font-weight="700">piano</text>
      <text x="62" y="82" text-anchor="middle" font-size="12" fill="${x1 > 0.5 ? 'white' : '#1f2937'}">${x1.toFixed(2)}</text>
      <circle cx="62" cy="176" r="24" fill="${nodeFill(x2)}" stroke="#1f2937" stroke-width="2"/>
      <text x="62" y="144" text-anchor="middle" font-size="12" fill="#1f2937" font-weight="700">homework</text>
      <text x="62" y="180" text-anchor="middle" font-size="12" fill="${x2 > 0.5 ? 'white' : '#1f2937'}">${x2.toFixed(2)}</text>

      <rect x="222" y="57" width="66" height="42" rx="6" fill="white" stroke="${edgeColor(c1)}" stroke-width="2"/>
      <text x="255" y="74" text-anchor="middle" font-size="11" fill="#1f2937">× ${w1.toFixed(2)}</text>
      <text x="255" y="91" text-anchor="middle" font-size="12" fill="${edgeColor(c1)}" font-weight="700">${c1.toFixed(2)}</text>
      <rect x="222" y="155" width="66" height="42" rx="6" fill="white" stroke="${edgeColor(c2)}" stroke-width="2"/>
      <text x="255" y="172" text-anchor="middle" font-size="11" fill="#1f2937">× ${w2.toFixed(2)}</text>
      <text x="255" y="189" text-anchor="middle" font-size="12" fill="${edgeColor(c2)}" font-weight="700">${c2.toFixed(2)}</text>
      <rect x="222" y="107" width="66" height="42" rx="6" fill="white" stroke="${edgeColor(b)}" stroke-width="2"/>
      <text x="255" y="124" text-anchor="middle" font-size="11" fill="#1f2937">bias</text>
      <text x="255" y="141" text-anchor="middle" font-size="12" fill="${edgeColor(b)}" font-weight="700">${b.toFixed(2)}</text>

      <circle cx="388" cy="126" r="28" fill="${nodeFill(z / 3)}" stroke="#1f2937" stroke-width="2"/>
      <text x="388" y="122" text-anchor="middle" font-size="13" fill="#1f2937" font-weight="700">sum</text>
      <text x="388" y="140" text-anchor="middle" font-size="13" fill="#1f2937">${z.toFixed(2)}</text>
      <rect x="444" y="101" width="72" height="50" rx="7" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>
      <text x="480" y="121" text-anchor="middle" font-size="12" fill="#1f2937" font-weight="700">sigmoid</text>
      <text x="480" y="139" text-anchor="middle" font-size="12" fill="#2563eb">${out.toFixed(2)}</text>
      <circle cx="558" cy="126" r="30" fill="${outFill}" stroke="${outStroke}" stroke-width="3"/>
      <text x="558" y="131" text-anchor="middle" font-size="15" fill="${outStroke}" font-weight="800">${decision}</text>

      <text x="310" y="232" text-anchor="middle" font-size="13" fill="#374151">weighted sum = ${c1.toFixed(2)} + ${c2.toFixed(2)} + ${b.toFixed(2)} = ${z.toFixed(2)}</text>
    `;
  }
  ids.forEach(id => document.getElementById(id).addEventListener('input', update));
  update();
})();

