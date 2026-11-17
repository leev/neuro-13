let pyReady = null;

function boot(base) {
  importScripts(base + 'pyodide.js');
  return loadPyodide({ indexURL: base }).then(py => {
    py.setStdout({ batched: s => postMessage({ type: 'out', text: s + '\n' }) });
    py.setStderr({ batched: s => postMessage({ type: 'err', text: s + '\n' }) });
    // Without a terminal attached Python buffers its output in large blocks,
    // so a training loop's progress would only appear when the script ends.
    py.runPython('import sys\nsys.stdout.reconfigure(line_buffering=True)\nsys.stderr.reconfigure(line_buffering=True)');
    return py;
  });
}

onmessage = async e => {
  const { code, base, files } = e.data;
  try {
    if (!pyReady) {
      postMessage({ type: 'status', text: 'loading Python…' });
      pyReady = boot(base);
    }
    const py = await pyReady;
    postMessage({ type: 'status', text: 'loading packages…' });
    await py.loadPackagesFromImports(code);
    for (const f of files) {
      postMessage({ type: 'status', text: 'downloading ' + f.name + '…' });
      const res = await fetch(f.url);
      if (!res.ok) throw new Error('could not download ' + f.name);
      py.FS.writeFile(f.name, new Uint8Array(await res.arrayBuffer()));
    }
    postMessage({ type: 'status', text: 'running…', ready: true });
    await py.runPythonAsync(code);
    postMessage({ type: 'status', text: 'Python ready', ready: true });
  } catch (err) {
    postMessage({ type: 'err', text: String((err && err.message) || err) + '\n' });
  }
  postMessage({ type: 'done' });
};
