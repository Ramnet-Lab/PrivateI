// Graph page. Plain vis-network against /api/graph; no framework, and the
// library is served from this container so the page works offline.

const TYPE_COLORS = {
  PERSON: '#4a9eff', ORG: '#f2b134', LOCATION: '#3fa06a',
  EVENT: '#c264d6', DOCUMENT: '#8c9099', CLAIM: '#d0453c',
};

const statusEl = document.getElementById('graphStatus');
const panel = document.getElementById('panel');
let network = null;
let allNodes = [], allEdges = [];

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
    title: `${e.predicate}\n${e.source_file || e.source_doc} page ${e.source_page}\n“${(e.quote || '').slice(0, 200)}”`,
  }));

  draw(allNodes, allEdges);
  statusEl.textContent = `${allNodes.length} entities, ${allEdges.length} connections`;
}

function draw(nodes, edges) {
  const container = document.getElementById('net');
  const nodeSet = new vis.DataSet(nodes);
  const edgeSet = new vis.DataSet(edges);
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
  network.on('click', params => {
    if (params.nodes.length) showEntity(params.nodes[0]);
  });
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
  const nodes = allNodes.filter(n =>
    (!type || n.entityType === type) &&
    (!term || n.label.toLowerCase().includes(term)));
  const keep = new Set(nodes.map(n => n.id));
  draw(nodes, allEdges.filter(e => keep.has(e.from) && keep.has(e.to)));
  document.getElementById('graphStatus').textContent =
    `${nodes.length} of ${allNodes.length} entities shown`;
}

document.getElementById('search').addEventListener('input', debounce(applyFilters, 250));
document.getElementById('typeFilter').addEventListener('change', applyFilters);
document.getElementById('refit').addEventListener('click', () => network && network.fit());

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

load();
