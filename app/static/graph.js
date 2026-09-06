// Graph page. Plain vis-network against /api/graph; no framework, and the
// library is served from this container so the page works offline.

const TYPE_COLORS = {
  PERSON: '#4a9eff', ORG: '#f2b134', LOCATION: '#3fa06a',
  EVENT: '#c264d6', DOCUMENT: '#8c9099', CLAIM: '#d0453c',
};

const statusEl = document.getElementById('graphStatus');
const panel = document.getElementById('panel');
let network = null;
let allNodes = [], allEdges = [], inferredEdges = [];

// Inferred edges are fetched separately and are off until asked for. /api/graph
// answers what the record says, and that answer must not change because someone
// ran a pass over it - so they are a second request, drawn differently, and the
// page looks exactly as it did before this existed until the box is ticked.
const INFERRED_COLOR = '#c264d6';
const EDGE_COLOR = '#3a4150';
const LIT_EDGE = '#7a869c';   // an evidence edge with focus on it

// Focusing a node. At 288 entities and nearly a thousand edges the whole graph
// is a hairball, and the only way to read one entity's neighbourhood is to push
// everything else back rather than to hunt for it. Dimmed rather than hidden:
// removing nodes would restabilise the physics and move the very thing being
// looked at out from under the cursor.
const DIM_NODE = 0.06;
const DIM_LABEL = 'rgba(230, 232, 236, 0.05)';
const DIM_EDGE = 'rgba(58, 65, 80, 0.05)';
let nodeSet = null, edgeSet = null, focused = null;

function nodeSize(degree) {
  return 12 + Math.min(26, Math.sqrt(degree || 1) * 7);
}

async function load() {
  statusEl.textContent = 'loading…';
  let data;
  try {
    data = await (await fetch('/api/graph')).json();
  } catch (err) {
    statusEl.textContent = 'could not load the graph';
    return;
  }
  if (data.error) { statusEl.textContent = data.error; return; }
  if (!data.nodes.length) {
    statusEl.textContent = 'nothing in the graph yet — upload a document first';
    return;
  }

  allNodes = data.nodes.map(n => ({
    id: n.id,
    label: n.name,
    title: `${n.type}: ${n.name}`,
    color: TYPE_COLORS[n.type] || '#8c9099',
    size: nodeSize(n.degree),
    entityType: n.type,
  }));
  allEdges = data.edges.map((e, i) => ({
    id: e.triple_id || ('e' + i),
    from: e.source,
    to: e.target,
    label: e.predicate,
    // Stated rather than left to the global style, because focusing dims every
    // edge and then has to put each one back the way it was.
    color: { color: EDGE_COLOR, highlight: '#4a9eff' },
    font: { color: '#939aa8' },
    title: `${e.predicate}\n${e.source_file || e.source_doc} page ${e.source_page}\n“${(e.quote || '').slice(0, 200)}”`,
  }));

  draw(allNodes, allEdges);
  statusEl.textContent = `${allNodes.length} entities, ${allEdges.length} connections`;
  loadInferred();
}

async function loadInferred() {
  let data;
  try {
    data = await (await fetch('/api/graph/inferred')).json();
  } catch (err) {
    return;                       // the evidence graph is drawn regardless
  }
  // Dashed, coloured and labelled with the relation, so an inference can never
  // be read off the screen as something a document said. The tooltip leads with
  // the word "inferred" and gives the basis instead of a quote, because there
  // is no quote - that is the whole difference.
  inferredEdges = (data.edges || []).map((e, i) => ({
    id: 'inferred-' + i,
    from: e.source,
    to: e.target,
    label: e.relation,
    dashes: true,
    width: 1,
    color: { color: INFERRED_COLOR, opacity: 0.75, highlight: '#e0a0f0' },
    font: { color: INFERRED_COLOR, size: 11 },
    inferredColor: true,
    title: `inferred — not stated in any document\n${e.relation}` +
           ` (confidence ${(e.confidence || 0).toFixed(2)})\n${e.basis || ''}`,
    inferred: true,
  }));
  const box = document.getElementById('showInferred');
  if (inferredEdges.length) {
    box.parentElement.title =
      `${inferredEdges.length} inferred connection(s). Not drawn from any ` +
      `document; a model's reading of two entities that were.`;
  } else {
    box.disabled = true;
    box.parentElement.title = 'No pass has been run yet — see the Links page.';
  }
}

function draw(nodes, edges) {
  const container = document.getElementById('net');
  focused = null;
  nodeSet = new vis.DataSet(nodes);
  edgeSet = new vis.DataSet(edges);
  network = new vis.Network(container, { nodes: nodeSet, edges: edgeSet }, {
    physics: {
      solver: 'forceAtlas2Based',
      forceAtlas2Based: { gravitationalConstant: -70, springLength: 140 },
      stabilization: { iterations: 220 },
    },
    interaction: { hover: true, tooltipDelay: 150, navigationButtons: true, keyboard: false },
    nodes: { shape: 'dot', font: { color: '#e6e8ec', size: 14 } },
    edges: {
      color: { color: '#3a4150', highlight: '#4a9eff' },
      font: { size: 10, color: '#939aa8', strokeWidth: 0 },
      arrows: { to: { scaleFactor: 0.5 } },
      smooth: { type: 'continuous' },
    },
  });
  // A pan is a click as far as the canvas is concerned - press, move, release,
  // and vis reports a click on empty space at the end of it. Releasing the
  // focus on that makes a focused neighbourhood impossible to move around,
  // which is the main thing anyone wants to do with one. So a drag is
  // remembered and the click that ends it is not treated as a click.
  let dragged = false;
  network.on('dragStart', () => { dragged = false; });
  network.on('dragging', () => { dragged = true; });

  network.on('click', params => {
    if (params.nodes.length) {
      focus(params.nodes[0]);
      showEntity(params.nodes[0]);
      dragged = false;
      return;
    }
    if (dragged) {
      dragged = false;      // the graph was moved, not dismissed
      return;
    }
    // A real click on empty space puts the whole graph back. Clicking an edge
    // counts as empty: the pair it joins is already lit by whichever end was
    // focused.
    focus(null);
  });
}

// Bring one node and everything it touches forward, and push the rest back until
// the next click. Called with null to restore.
function focus(id) {
  if (!nodeSet || !edgeSet) return;
  focused = id;
  const edges = edgeSet.get();

  if (id === null) {
    nodeSet.update(nodeSet.get().map(n => ({
      id: n.id, opacity: 1, font: { color: '#e6e8ec' },
    })));
    edgeSet.update(edges.map(e => ({
      id: e.id,
      color: e.inferredColor
        ? { color: INFERRED_COLOR, opacity: 0.75, highlight: '#e0a0f0' }
        : { color: EDGE_COLOR, highlight: '#4a9eff' },
      font: { color: e.inferredColor ? INFERRED_COLOR : '#939aa8' },
    })));
    return;
  }

  const near = new Set([id]);
  const touching = new Set();
  for (const e of edges) {
    if (e.from === id || e.to === id) {
      touching.add(e.id);
      near.add(e.from);
      near.add(e.to);
    }
  }
  nodeSet.update(nodeSet.get().map(n => ({
    id: n.id,
    opacity: near.has(n.id) ? 1 : DIM_NODE,
    font: { color: near.has(n.id) ? '#e6e8ec' : DIM_LABEL },
  })));
  edgeSet.update(edges.map(e => ({
    id: e.id,
    color: touching.has(e.id)
      ? (e.inferredColor
          ? { color: INFERRED_COLOR, opacity: 1, highlight: '#e0a0f0' }
          : { color: LIT_EDGE, highlight: '#4a9eff' })
      : { color: DIM_EDGE, opacity: 0.05 },
    font: { color: touching.has(e.id)
      ? (e.inferredColor ? INFERRED_COLOR : '#c3c9d4') : DIM_LABEL },
  })));
}

async function showEntity(id) {
  panel.innerHTML = '<p class="panel-empty">loading…</p>';
  let detail;
  try {
    detail = await (await fetch('/api/entity/' + encodeURIComponent(id))).json();
  } catch (err) {
    panel.innerHTML = '<p class="panel-empty">could not load that entity</p>';
    return;
  }

  const facts = detail.facts.map(f => {
    const phrase = f.direction === 'out'
      ? `${escapeHtml(detail.name)} <em>${escapeHtml(f.predicate)}</em> <strong>${escapeHtml(f.other_name)}</strong>`
      : `<strong>${escapeHtml(f.other_name)}</strong> <em>${escapeHtml(f.predicate)}</em> ${escapeHtml(detail.name)}`;
    return `<li>
      <div class="fact">${phrase}${f.event_date ? ` <span class="small">${escapeHtml(f.event_date)}</span>` : ''}</div>
      <blockquote>“${escapeHtml(f.quote || '')}”</blockquote>
      <a class="small" href="/documents/${encodeURIComponent(f.source_doc)}#page-${f.source_page}" target="_blank">
        ${escapeHtml(f.source_file || f.source_doc)} · page ${f.source_page}</a>
    </li>`;
  }).join('');

  panel.innerHTML = `
    <h2>${escapeHtml(detail.name)}</h2>
    <p class="small">${escapeHtml(detail.type)} · ${detail.facts.length} connection(s)</p>
    <ul class="facts">${facts || '<li class="panel-empty">no connections</li>'}</ul>`;
}

function escapeHtml(value) {
  const div = document.createElement('div');
  div.textContent = value == null ? '' : String(value);
  return div.innerHTML;
}

function applyFilters() {
  const term = document.getElementById('search').value.trim().toLowerCase();
  const type = document.getElementById('typeFilter').value;
  const withInferred = document.getElementById('showInferred').checked;
  const nodes = allNodes.filter(n =>
    (!type || n.entityType === type) &&
    (!term || n.label.toLowerCase().includes(term)));
  const keep = new Set(nodes.map(n => n.id));
  const edges = allEdges.concat(withInferred ? inferredEdges : [])
    .filter(e => keep.has(e.from) && keep.has(e.to));
  draw(nodes, edges);
  const shown = withInferred
    ? `${nodes.length} of ${allNodes.length} entities shown, ` +
      `${edges.length - allEdges.filter(e => keep.has(e.from) && keep.has(e.to)).length}` +
      ` inferred connection(s) included`
    : `${nodes.length} of ${allNodes.length} entities shown`;
  document.getElementById('graphStatus').textContent = shown;
}

document.getElementById('search').addEventListener('input', debounce(applyFilters, 250));
document.getElementById('typeFilter').addEventListener('change', applyFilters);
document.getElementById('showInferred').addEventListener('change', applyFilters);
// A second way out of a focus, for anyone who does not want to hunt for empty
// canvas in a graph this dense.
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape' && focused !== null) focus(null);
});
document.getElementById('refit').addEventListener('click', () => network && network.fit());

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

load();
