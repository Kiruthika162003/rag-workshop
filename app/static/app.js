// Four tabs, one scenario each. Ask sends the question to every card in the tab at once;
// each card fills in when its answer is back. Explain opens the side panel for the tab.

const viewer = {
  title: document.getElementById('viewer-title'),
  meta: document.getElementById('viewer-meta'),
  body: document.getElementById('viewer-body'),
};
const drawer = document.getElementById('drawer');

const escape = text => String(text).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const each = (selector, work) => document.querySelectorAll(selector).forEach(work);

function choose(stage) {
  each('.tab', tab => tab.setAttribute('aria-selected', tab.dataset.stage === stage));
  each('.stage', section => { section.hidden = section.dataset.stage !== stage; });
  each('.explained', part => { part.hidden = part.dataset.stage !== stage; });
}

each('.tab', tab => tab.addEventListener('click', () => choose(tab.dataset.stage)));
each('.explain', button => button.addEventListener('click', () => { drawer.hidden = false; }));
document.getElementById('close').addEventListener('click', () => { drawer.hidden = true; });
document.addEventListener('keydown', event => { if (event.key === 'Escape') drawer.hidden = true; });
choose(document.querySelector('.tab').dataset.stage);

each('.stage', section => {
  const box = section.querySelector('textarea');
  const askAll = () => box.value.trim() && section.querySelectorAll('.card').forEach(card => ask(card, box.value.trim()));
  section.querySelector('.ask').addEventListener('submit', event => { event.preventDefault(); askAll(); });
  box.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); askAll(); } });
});
each('.card', card => card.querySelector('.retry').addEventListener('click', () => ask(card, card.dataset.question)));

const WAIT = 60_000;  // a card never spins longer than this; it offers Retry instead

async function ask(card, question) {
  const asked = String(Date.now() + Math.random());
  Object.assign(card.dataset, { question, asked });
  const answer = card.querySelector('.answer');
  answer.className = 'answer working';
  answer.innerHTML = '';
  card.querySelector('.retry').hidden = true;
  card.querySelector('.verdict').textContent = '';
  card.querySelector('.extra').innerHTML = '';
  card.querySelector('.sources').innerHTML = '';
  card.querySelector('.metrics').innerHTML = '';
  try {
    const reply = await fetch('/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: card.dataset.kind, question }), signal: AbortSignal.timeout(WAIT),
    });
    const result = await reply.json();
    if (card.dataset.asked !== asked) return;  // a newer question was asked meanwhile
    if (!reply.ok) throw new Error(result.error);
    show(card, result);
  } catch (error) {
    if (card.dataset.asked !== asked) return;
    answer.className = 'answer';
    answer.textContent = error.name === 'TimeoutError'
      ? `No answer after ${WAIT / 1000} seconds. The model may be busy: try again.`
      : 'This one did not answer: ' + error.message;
    card.querySelector('.retry').hidden = false;
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
  if (result.note) answer.insertAdjacentHTML('beforeend', `<p class="note">${escape(result.note)}</p>`);
  card.querySelector('.metrics').innerHTML = '<button type="button" class="evaluate">Evaluate this answer</button>';
  card.querySelector('.evaluate').addEventListener('click', () => evaluate(card, result));

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

// The four RAG metrics, judged by a model, only when asked for: each one costs a Gemini call.
const METRICS = [
  ['faithfulness', 'Faithfulness', 'claims in the answer that the sources support'],
  ['answer_relevancy', 'Answer relevancy', 'how directly it answers the question asked'],
  ['context_precision', 'Context precision', 'useful sources retrieved, and ranked first'],
  ['context_recall', 'Context recall', 'the reference answer, found in the sources'],
];

async function evaluate(card, result) {
  const box = card.querySelector('.metrics');
  const asked = card.dataset.asked;
  box.innerHTML = '<p class="muted">Judging...</p>';
  try {
    const reply = await fetch('/evaluate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: AbortSignal.timeout(WAIT),
      body: JSON.stringify({ question: card.dataset.question, answer: result.answer, sources: result.sources }),
    });
    const scores = await reply.json();
    if (card.dataset.asked !== asked) return;
    if (!reply.ok) throw new Error(scores.error);
    box.innerHTML = METRICS.map(([key, name, about]) => {
      if (!scores[key]) return `<div class="metric off" title="${about}"><span>${name}</span><i></i><b>no reference</b></div>`;
      const value = scores[key].score;
      const level = value >= .8 ? 'high' : value >= .5 ? 'mid' : 'low';
      return `<div class="metric ${level}" title="${about}"><span>${name}</span><i><u style="width:${value * 100}%"></u></i><b>${value.toFixed(2)}</b></div>`;
    }).join('') + details(scores);
  } catch (error) {
    if (card.dataset.asked !== asked) return;
    box.innerHTML = `<p class="muted">The judge did not answer: ${escape(error.message)}</p><button type="button" class="evaluate">Try again</button>`;
    box.querySelector('.evaluate').addEventListener('click', () => evaluate(card, result));
  }
}

function details(scores) {
  const line = (ok, text) => `<li class="${ok ? 'ok' : 'no'}">${ok ? '✓' : '✗'} ${escape(text)}</li>`;
  const claims = scores.faithfulness.claims.map(c => line(c.supported, c.claim)).join('');
  const found = scores.context_recall ? scores.context_recall.statements.map(s => line(s.found, s.statement)).join('') : '';
  return `<details><summary>Why these scores</summary>
    <p>Claims in the answer:</p><ul>${claims}</ul>
    ${found ? `<p>The reference answer, in the sources?</p><ul>${found}</ul>` : ''}
    <p>Useful sources, in retrieved order: ${scores.context_precision.useful.map(u => u ? '✓' : '✗').join(' ') || 'none'}</p>
    <p>Questions this answer answers: ${scores.answer_relevancy.questions.map(escape).join(' · ') || 'none'}</p></details>`;
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
  document.getElementById('viewer').scrollIntoView({ behavior: 'smooth', block: 'start' });
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
