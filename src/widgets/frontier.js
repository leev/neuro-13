(function toolAppWidget() {
  const win = document.getElementById('ta-window');
  if (!win) return;
  const byId = id => document.getElementById(id);
  const goBtn = byId('ta-go'), note = byId('ta-note');
  const WEEKS = 3;
  const STALE_YEAR = 2020;
  const LIBRARY = {
    volcanoes: [
      { title: 'Rivers of Fire', pages: 412, added: 2025 },
      { title: 'The Sleeping Mountain', pages: 268, added: 2011 },
      { title: 'Volcano Field Notes', pages: 156, added: 2017 },
    ],
    sharks: [
      { title: 'The Quiet Hunters', pages: 198, added: 2009 },
      { title: 'Shark Atlas', pages: 344, added: 2025 },
      { title: 'Teeth of the Deep', pages: 230, added: 2016 },
    ],
    space: [
      { title: 'A Night Sky Diary', pages: 144, added: 2014 },
      { title: 'Next Stop, Saturn', pages: 301, added: 2019 },
      { title: 'Every Planet We Know', pages: 520, added: 2025 },
    ],
  };
  const SYSTEM = 'You help students at Maple Street School find books.\n' +
    'You have two tools:\n' +
    '  search_catalog: lists the library\'s books on a topic\n' +
    '  calculator: works out a piece of arithmetic\n' +
    'To use one, reply with only a request, like\n' +
    '  {"tool": "calculator", "expression": "2 + 2"}';

  let topic = 'volcanoes', catalog = 'fresh', turns = [], trips = 0, toolRuns = 0, pending = null, finished = false;

  const spaced = value => JSON.stringify(value).replace(/":/g, '": ').replace(/,"/g, ', "');

  const TOOLS = {
    search_catalog(request) {
      const books = LIBRARY[request.topic].filter(b => catalog === 'fresh' || b.added <= STALE_YEAR);
      return '[' + books.map(b => spaced({ title: b.title, pages: b.pages })).join(',\n ') + ']';
    },
    calculator(request) {
      const sum = /^(\d+) \/ (\d+)$/.exec(request.expression);
      return String(+sum[1] / +sum[2]);
    },
  };

  // The stand-in for the model is handed the window and nothing else, the
  // same rule a real model lives by: whatever it "knows" it has to read there.
  function scriptedModel(windowTurns) {
    const question = windowTurns.find(t => t.tag === 'user').text;
    const results = windowTurns.filter(t => t.tag === 'tool').map(t => t.text);
    const days = 7 * +/(\d+) weeks/.exec(question)[1];
    if (results.length === 0) {
      return { call: { tool: 'search_catalog', topic: /book about (\w+)/.exec(question)[1] } };
    }
    const longest = JSON.parse(results[0]).reduce((a, b) => (b.pages > a.pages ? b : a));
    if (results.length === 1) {
      return { call: { tool: 'calculator', expression: `${longest.pages} / ${days}` } };
    }
    const perDay = Math.ceil(+results[1]);
    return { answer: `The longest book about ${/book about (\w+)/.exec(question)[1]} in the catalog is ${longest.title}, at ${longest.pages} pages. ` +
      `Read ${perDay} pages a day and you will finish it inside your ${days} days.` };
  }

  function add(tag, who, text) {
    turns.push({ tag, who, text, fresh: true });
  }

  function render() {
    win.innerHTML = '';
    turns.forEach(t => {
      const row = document.createElement('div');
      row.className = 'fr-turn' + (t.fresh ? ' fr-new' : '');
      row.dataset.who = t.who;
      const tag = document.createElement('span');
      tag.className = 'fr-tag';
      tag.textContent = `<${t.tag}>`;
      const body = document.createElement('p');
      body.textContent = t.text;
      row.append(tag, body);
      win.appendChild(row);
      t.fresh = false;
    });
    byId('ta-trips').textContent = trips;
    byId('ta-tools').textContent = toolRuns;
    byId('ta-lines').textContent = turns.length;
    goBtn.disabled = finished;
    goBtn.textContent = finished ? 'the loop is finished' : pending ? `run the tool: ${pending.tool}` : 'send the window to the model';
  }

  function say(text, wrong) {
    note.textContent = text;
    note.classList.toggle('ta-wrong', !!wrong);
  }

  function reset() {
    turns = [];
    trips = 0;
    toolRuns = 0;
    pending = null;
    finished = false;
    add('system', 'app', SYSTEM);
    add('user', 'student', `I have ${WEEKS} weeks to read the longest book about ${topic} in our library. How many pages a day is that?`);
    turns.forEach(t => { t.fresh = false; });
    say('The window holds two things: the app\'s instructions and the student\'s question. Nothing has run yet. You are the app, so send the window to the model.');
    render();
  }

  function runModel() {
    const out = scriptedModel(turns);
    trips++;
    if (out.call) {
      pending = out.call;
      add('assistant', 'model', spaced(out.call));
      say(trips === 1
        ? 'The model did not answer. It wrote a request, in the format the instructions asked for. That is all a tool call is: text. The model cannot open the catalog. You can.'
        : 'The model read the whole window again, picked the longest book out of your search result, and now wants a division done. It is asking instead of guessing.');
      return;
    }
    finished = true;
    add('assistant', 'model', out.answer);
    const truth = LIBRARY[topic].reduce((a, b) => (b.pages > a.pages ? b : a));
    if (catalog === 'stale') {
      say(`The model did everything right and the answer is wrong. The library's longest book about ${topic} is ${truth.title} (${truth.pages} pages), added in ${truth.added}. ` +
        'The 2020 copy of the catalog does not list it, and nothing in the window could have told the model so.', true);
    } else {
      say(`No request this time. Everything the model needed was in the window, so it wrote an answer, and the loop is over. It ran ${trips} times and never touched the catalog or the calculator. Now try the 2020 catalog.`);
    }
  }

  function runTool() {
    const result = TOOLS[pending.tool](pending);
    toolRuns++;
    add('tool', 'app', result);
    say(pending.tool === 'calculator'
      ? 'Your browser really did that division just now, and you pasted the result into the window. Send it one more time.'
      : 'You searched the catalog with ordinary code (no neural network needed) and pasted what came back into the window. The model has not seen it yet. Send the window again.');
    pending = null;
  }

  goBtn.addEventListener('click', () => {
    if (finished) return;
    const before = goBtn.getBoundingClientRect().top;
    if (pending) runTool(); else runModel();
    render();
    // The window grows above the button on every press. Scrolling by the same
    // amount keeps the button under the finger, so five presses need no aiming.
    window.scrollBy({ top: goBtn.getBoundingClientRect().top - before, behavior: 'instant' });
  });
  byId('ta-reset').addEventListener('click', reset);

  function wire(groupId, key, apply) {
    const group = byId(groupId);
    const options = [...group.querySelectorAll('.toggle')];
    const pick = el => {
      options.forEach(o => {
        o.classList.toggle('on', o === el);
        o.setAttribute('aria-pressed', o === el);
      });
      apply(el.dataset[key]);
      reset();
    };
    options.forEach(el => {
      el.addEventListener('click', () => pick(el));
      el.addEventListener('keydown', ev => {
        if (ev.key !== 'Enter' && ev.key !== ' ') return;
        ev.preventDefault();
        pick(el);
      });
    });
  }
  wire('ta-topic', 'topic', v => { topic = v; });
  wire('ta-catalog', 'catalog', v => { catalog = v; });
  reset();
})();
