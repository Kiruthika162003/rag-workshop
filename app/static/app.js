// One question goes to all six kinds of RAG at once. Each card fills in when its answer is back.

const form = document.getElementById('ask');
const input = document.getElementById('question');
const cards = [...document.querySelectorAll('.card')];
const viewer = {
  title: document.getElementById('viewer-title'),
  meta: document.getElementById('viewer-meta'),
  body: document.getElementById('viewer-body'),
};

const escape = text => String(text).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

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
    const result = await reply.json();
    if (!reply.ok) throw new Error(result.error);
    show(card, result);
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
    cite.addEventListener('click', () => open(result.sources[cite.dataset.n])));

  const verdict = card.querySelector('.verdict');
  verdict.textContent = `${result.seconds}s` + (result.right === null ? '' : result.right ? '  ✓ right' : '  ✗ wrong');
  verdict.className = 'verdict ' + (result.right === null ? '' : result.right ? 'right' : 'wrong');

  const extra = card.querySelector('.extra');
  if (result.steps) extra.innerHTML = result.steps.map(s => `<div class="step">${escape(s.tool)}(${escape(Object.values(s.args).join(', '))})</div>`).join('');
  if (result.attempts) extra.innerHTML = result.attempts.map(a => `<div class="step">search(${escape(a.search.join(' | '))}) best ${a.best}/10</div>`).join('');
  if (result.graph) drawGraph(extra, result.graph.path);

  // Retrieved is not the same as used: show what the answer cites, fold away the rest.
  const cited = new Set([...answer.querySelectorAll('.cite')].map(c => Number(c.dataset.n)));
  const chip = (s, n) => `<span class="source" data-n="${n}">${result.steps ? s.id : n + 1} · ${s.type} · page ${s.page}</span>`;
  const used = result.sources.map(chip).filter((_, n) => cited.has(n));
  const unused = result.sources.map(chip).filter((_, n) => !cited.has(n));
  card.querySelector('.sources').innerHTML = used.join('') +
    (unused.length ? `<details><summary>+${unused.length} retrieved but not cited</summary>${unused.join('')}</details>` : '');
  card.querySelectorAll('.source').forEach(s => s.addEventListener('click', () => open(result.sources[s.dataset.n])));
}

// Citations look like [2] and [1, 3] or, from the agent, [t14] and [p2].
function withCitations(text, sources) {
  const html = escape(text).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/^\* /gm, '• ').replace(/\n/g, '<br>');
  return html.replace(/\[([^\]]+)\]/g, (whole, inside) => {
    const chips = inside.split(',').map(s => s.trim())
      .map(item => ({ label: item, n: /^\d+$/.test(item) ? Number(item) - 1 : sources.findIndex(s => s.id === item) }))
      .filter(c => sources[c.n]);
    return chips.length ? chips.map(c => `<span class="cite" data-n="${c.n}">${c.label}</span>`).join(' ') : whole;
  });
}

function open(source) {
  viewer.title.textContent = source.type === 'picture' ? `Figure or table, page ${source.page}` : `Page ${source.page}`;
  viewer.meta.textContent = source.section || '';
  viewer.body.innerHTML = `<img src="/page/${source.id}" alt="Page ${source.page}, with the passage highlighted">
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
