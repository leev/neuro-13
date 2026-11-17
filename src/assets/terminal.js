(function pythonTerminals() {
  const byId = id => document.getElementById(id);
  const editors = new Map();
  let worker = null, running = null;

  document.querySelectorAll('.code-editor').forEach(ta => {
    const original = ta.value;
    let cm = null;
    if (window.CodeMirror) {
      cm = CodeMirror.fromTextArea(ta, {
        mode: 'python', theme: 'material-darker', lineNumbers: true,
        indentUnit: 4, lineWrapping: false, readOnly: ta.readOnly,
        extraKeys: { Tab: editor => editor.execCommand('indentMore'), 'Shift-Tab': editor => editor.execCommand('indentLess') },
      });
      cm.setSize(null, Math.min(520, Math.max(220, original.split('\n').length * 19 + 16)));
    }
    editors.set(ta.id, {
      original,
      files: (ta.dataset.files || '').split(',').filter(Boolean),
      get: () => (cm ? cm.getValue() : ta.value),
      set: value => { if (cm) cm.setValue(value); else ta.value = value; },
    });
  });

  function append(el, text, cls) {
    const span = document.createElement('span');
    if (cls) span.className = cls;
    span.textContent = text;
    el.appendChild(span);
    el.scrollTop = el.scrollHeight;
  }

  function finish() {
    if (!running) return;
    running.btn.textContent = running.label;
    running.btn.classList.remove('stop');
    document.querySelectorAll('[data-py]').forEach(b => { b.disabled = false; });
    running = null;
  }

  function run(btn) {
    const id = btn.getAttribute('data-py');
    const editor = editors.get('py-code-' + id);
    const out = byId('py-out-' + id), status = byId('py-status-' + id);
    out.textContent = '';
    append(out, '$ python3\n', 'prompt');
    running = { id, btn, out, label: btn.textContent };
    document.querySelectorAll('[data-py]').forEach(b => { b.disabled = b !== btn; });
    btn.textContent = '■ stop';
    btn.classList.add('stop');
    if (!worker) worker = new Worker('assets/py-worker.js');
    worker.onmessage = e => {
      const m = e.data;
      if (m.type === 'out') append(out, m.text);
      else if (m.type === 'err') append(out, m.text, 'err');
      else if (m.type === 'status' && status) {
        status.textContent = m.text;
        status.classList.toggle('ready', !!m.ready);
      } else if (m.type === 'done') finish();
    };
    worker.onerror = e => {
      append(out, 'Python could not start: ' + (e.message || 'check your internet connection') + '\n', 'err');
      worker.terminate();
      worker = null;
      finish();
    };
    worker.postMessage({
      code: editor.get(),
      base: window.PYODIDE_URL,
      files: editor.files.map(name => ({ name, url: new URL('py/' + name, location.href).href })),
    });
  }

  function stop() {
    worker.terminate();
    worker = null;
    append(running.out, '\n(stopped)\n', 'dim');
    const status = byId('py-status-' + running.id);
    if (status) { status.textContent = 'stopped'; status.classList.remove('ready'); }
    finish();
  }

  document.addEventListener('click', e => {
    const runBtn = e.target.closest('[data-py]');
    if (runBtn) {
      if (running && running.btn === runBtn) stop(); else if (!running) run(runBtn);
      return;
    }
    const resetBtn = e.target.closest('[data-py-reset]');
    if (resetBtn) {
      const id = resetBtn.getAttribute('data-py-reset');
      const editor = editors.get('py-code-' + id);
      editor.set(editor.original);
      byId('py-out-' + id).innerHTML = '<span class="dim">(code reset — click run)</span>';
    }
  });
})();
