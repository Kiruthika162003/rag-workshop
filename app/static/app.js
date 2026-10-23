// One question goes to all five kinds of RAG at once. Each card fills in when its answer is back.

const form = document.getElementById('ask');
const input = document.getElementById('question');
const cards = [...document.querySelectorAll('.card')];
const viewer = {
  title: document.getElementById('viewer-title'),
  meta: document.getElementById('viewer-meta'),
  body: document.getElementById('viewer-body'),
};

const escape = text => String(text).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const seconds = time => { const [m, s] = time.split(':').map(Number); return m * 60 + s; };

form.addEventListener('submit', event => { event.preventDefault(); askAll(input.value.trim()); });
document.querySelectorAll('#examples .chip').forEach(chip =>
  chip.addEventListener('click', () => { input.value = chip.dataset.question; askAll(chip.dataset.question); }));

function askAll(question) {
  if (!question) return;
  cards.forEach(card => ask(card, question));
}

async function ask(card, question) {
  const answer = card.querySelector('.answer');
  answer.className = 'answer working';
  answer.innerHTML = '';
  card.querySelector('.verdict').textContent = '';
  card.querySelector('.extra').innerHTML = '';
  card.querySelector('.sources').innerHTML = '';
  try {
    const reply = await fetch('/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: card.dataset.kind, question }),
    });
    show(card, await reply.json());
  } catch (error) {
    answer.className = 'answer';
    answer.textContent = 'This one did not answer: ' + error.message;
  }
}

function show(card, result) {
  const answer = card.querySelector('.answer');
  answer.className = 'answer';
  answer.innerHTML = withCitations(result.answer, result.sources);
  answer.querySelectorAll('.cite').forEach(cite =>
    cite.addEventListener('click', () => open(result.sources[cite.dataset.n], cite.dataset.time)));

  const verdict = card.querySelector('.verdict');
  verdict.textContent = `${result.seconds}s` + (result.right === null ? '' : result.right ? '  ✓ right' : '  ✗ wrong');
  verdict.className = 'verdict ' + (result.right === null ? '' : result.right ? 'right' : 'wrong');

  const extra = card.querySelector('.extra');
  if (result.steps) extra.innerHTML = result.steps.map(s => `<div class="step">${escape(s.tool)}(${escape(Object.values(s.args).join(', '))})</div>`).join('');
  if (result.graph) drawGraph(extra, result.graph.path);

  // Retrieved is not the same as used: show what the answer cites, fold away the rest.
  const cited = new Set([...answer.querySelectorAll('.cite')].map(c => Number(c.dataset.n)));
  const chip = (s, n) => `<span class="source" data-n="${n}">${result.steps ? s.id : n + 1} · ${s.type} · ${escape(s.title)}</span>`;
  const used = result.sources.map(chip).filter((_, n) => cited.has(n));
  const unused = result.sources.map(chip).filter((_, n) => !cited.has(n));
  card.querySelector('.sources').innerHTML = used.join('') +
    (unused.length ? `<details><summary>+${unused.length} retrieved but not cited</summary>${unused.join('')}</details>` : '');
  card.querySelectorAll('.source').forEach(s => s.addEventListener('click', () => open(result.sources[s.dataset.n])));
}

// Citations look like [2], [1, 3], [3, 0:12] or, from the agent, [t14] and [m4, 0:23].
function withCitations(text, sources) {
  const html = escape(text).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/^\* /gm, '• ').replace(/\n/g, '<br>');
  return html.replace(/\[([^\]]+)\]/g, (whole, inside) => {
    const items = inside.split(',').map(s => s.trim());
    const chips = [];
    items.forEach(item => {
      if (/^\d+:\d\d$/.test(item) && chips.length) { chips[chips.length - 1].time = item; return; }
      const n = /^\d+$/.test(item) ? Number(item) - 1 : sources.findIndex(s => s.id === item);
      if (sources[n]) chips.push({ n, label: item });
    });
    if (!chips.length) return whole;
    return chips.map(c => `<span class="cite" data-n="${c.n}"${c.time ? ` data-time="${c.time}"` : ''}>${c.label}${c.time ? ' ' + c.time : ''}</span>`).join(' ');
  });
}

function open(source, time) {
  viewer.title.textContent = source.title;
  viewer.meta.textContent = `${source.kind || ''} · dated ${source.date} · ${source.type}` + (source.type === 'text' || source.type === 'picture' ? ` · page ${source.page}` : '');
  if (source.type === 'audio' || source.type === 'video') {
    const start = time ? seconds(time) : 0;
    viewer.body.innerHTML = `<${source.type} controls src="/files/${source.doc}#t=${start}"></${source.type}>
      <div class="transcript">${escape(source.text).replace(/\[(\d+:\d\d)\]/g, '<span class="time" data-t="$1">[$1]</span>')}</div>`;
    const player = viewer.body.querySelector(source.type);
    viewer.body.querySelectorAll('.time').forEach(t => t.addEventListener('click', () => { player.currentTime = seconds(t.dataset.t); player.play(); }));
    if (time) player.play().catch(() => {});
    return;
  }
  viewer.body.innerHTML = `<img src="/page/${source.id}" alt="Page ${source.page} of ${escape(source.title)}, with the passage highlighted">
    <a href="/files/${source.doc}#page=${source.page}" target="_blank">Open the PDF at page ${source.page}</a>`;
}

// The facts the graph answer walked along, drawn as a little graph.
function drawGraph(where, facts) {
  if (!facts.length || !window.cytoscape) return;
  const box = document.createElement('div');
  box.className = 'graph-view';
  where.appendChild(box);
  const nodes = [...new Set(facts.flatMap(f => [f.subject, f.object]))].map(id => ({ data: { id } }));
  const edges = facts.map((f, i) => ({ data: { id: 'e' + i, source: f.subject, target: f.object, label: f.relation + (f.since ? ` (${f.since})` : '') } }));
  cytoscape({
    container: box, elements: [...nodes, ...edges], layout: { name: 'cose', animate: false, padding: 18 },
    userZoomingEnabled: false, maxZoom: 1.3,
    style: [
      { selector: 'node', style: { label: 'data(id)', 'font-size': 9, 'background-color': '#188038', width: 12, height: 12, 'text-valign': 'top', color: '#202124' } },
      { selector: 'edge', style: { label: 'data(label)', 'font-size': 7, width: 1.2, 'line-color': '#9aa0a6', 'target-arrow-color': '#9aa0a6',
                                   'target-arrow-shape': 'triangle', 'curve-style': 'bezier', color: '#5f6368', 'text-rotation': 'autorotate' } },
    ],
  });
}
