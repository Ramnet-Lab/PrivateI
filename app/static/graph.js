// Graph page. Plain vis-network against /api/graph; no framework, and the
// library is served from this container so the page works offline.

const TYPE_COLORS = {
  PERSON: '#4a9eff', ORG: '#f2b134', LOCATION: '#3fa06a',
  EVENT: '#c264d6', DOCUMENT: '#8c9099', CLAIM: '#d0453c',
};
const OTHER_COLOR = '#8c9099';
// An entity with no type and a predicate with no wording are both things the
// corpus can actually contain. They get a visible bucket rather than an empty
// checkbox label, because an unlabelled row is indistinguishable from a bug.
const UNTYPED = '(no type)';
const UNWORDED = '(no wording)';

const statusEl = document.getElementById('graphStatus');
const searchEl = document.getElementById('search');
const panel = document.getElementById('panel');
const wrap = document.getElementById('graphWrap');
const banner = document.getElementById('inferenceOnly');
let network = null;
let allNodes = [], allEdges = [], inferredEdges = [];

// Inferred edges are fetched separately and are off until asked for. /api/graph
// answers what the record says, and that answer must not change because someone
// ran a pass over it - so they are a second request, drawn differently, and the
// page looks exactly as it did before this existed until a box is ticked.
// INFERRED_COLOR is also spelled as --inferred in app.css, which draws the
// warning ring; the two have to stay equal.
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

// --- what is ticked ---------------------------------------------------------
// Three independent sets, and the stated/inferred split is a split in the state
// and not only in the layout. There is no operation anywhere below that can put
// a relation a model invented into the set that holds what a document said.
const selected = { type: new Set(), stated: new Set(), inferred: new Set() };
const order = { type: [], stated: [], inferred: [] };   // by count, descending
const counts = { type: new Map(), stated: new Map(), inferred: new Map() };
const subHeads = {};          // kind -> { box, keys } for the "occurs once" row
let hideOrphans = false;

// The previous hidden state of every node and edge, so a filter change pushes
// only what actually changed into vis rather than 462 no-op updates a keystroke.
const nodeHidden = new Map(), edgeHidden = new Map();

// Whether the inferred edges have reached the DataSet. update() on an id vis has
// never seen does not fail, it inserts - so a sweep that ran first would put 183
// edges consisting of an id and a hidden flag, with no endpoints, into the graph.
let inferredLive = false, inferredLoaded = false;

// At least this many one-off kinds, alongside at least one recurring kind,
// before the list is worth splitting. Below that the sub-heading costs more
// attention than it saves - which is why the 13 inferred relations render flat
// and the 39 stated predicates do not, and that difference is itself true.
const SPLIT_SINGLETONS_AT = 4;

function nodeSize(degree) {
  return 12 + Math.min(26, Math.sqrt(degree || 1) * 7);
}

function tally(map, key) { map.set(key, (map.get(key) || 0) + 1); }

function byCount(map) {
  return [...map.keys()].sort((a, b) => (map.get(b) - map.get(a)) || a.localeCompare(b));
}

// --- loading ----------------------------------------------------------------

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
    color: TYPE_COLORS[n.type] || OTHER_COLOR,
    size: nodeSize(n.degree),
    entityType: n.type || UNTYPED,
  }));
  allEdges = data.edges.map((e, i) => ({
    id: e.triple_id || ('e' + i),
    from: e.source,
    to: e.target,
    label: e.predicate,
    kind: e.predicate || UNWORDED,     // what the Links filter keys on
    // Stated rather than left to the global style, because focusing dims every
    // edge and then has to put each one back the way it was.
    color: { color: EDGE_COLOR, highlight: '#4a9eff' },
    font: { color: '#939aa8' },
    title: `${e.predicate}\n${e.source_file || e.source_doc} page ${e.source_page}\n“${(e.quote || '').slice(0, 200)}”`,
  }));
  // The DataSet is built once now and lives for the page, so a repeated
  // triple_id throws inside the constructor and blanks the graph rather than
  // costing one edge. Dropped here, where it is one line.
  const seenIds = new Set();
  allEdges = allEdges.filter(e => !seenIds.has(e.id) && seenIds.add(e.id));

  for (const n of allNodes) tally(counts.type, n.entityType);
  for (const e of allEdges) tally(counts.stated, e.kind);
  order.type = byCount(counts.type);
  order.stated = byCount(counts.stated);
  selected.type = new Set(order.type);       // everything the corpus has
  selected.stated = new Set(order.stated);   // every wording a document used

  buildTypeList();
  buildLinkList();
  build();
  applyFilters();
  loadInferred();
}

async function loadInferred() {
  let data;
  try {
    data = await (await fetch('/api/graph/inferred')).json();
  } catch (err) {
    inferredLoaded = true;      // the evidence graph is drawn regardless
    buildLinkList();
    applyFilters();
    return;
  }
  // Dashed, coloured and labelled with the relation, so an inference can never
  // be read off the screen as something a document said. The tooltip leads with
  // the word "inferred" and gives the basis instead of a quote, because there
  // is no quote - that is the whole difference.
  const known = new Set(allNodes.map(n => n.id));
  inferredEdges = (data.edges || [])
    // An inference can name an entity that is not on this canvas. Adding an edge
    // whose endpoint vis has never seen throws, and takes the whole batch with it.
    .filter(e => known.has(e.source) && known.has(e.target))
    .map((e, i) => ({
      id: 'inferred-' + i,
      from: e.source,
      to: e.target,
      label: e.relation,
      kind: e.relation || UNWORDED,
      dashes: true,
      width: 1,
      // The load-bearing line in this file. vis-network builds its simulation
      // from `true === options.physics` on each node and edge and never looks at
      // `hidden`, so an inferred edge left in the solver would pull the evidence
      // layout around whether it was drawn or not - and two entities dragged
      // adjacent by a model's guess look, on screen, like two entities the
      // documents connected. Switched off here, the 183 of them are drawn over
      // the record and can never move it.
      physics: false,
      color: { color: INFERRED_COLOR, opacity: 0.75, highlight: '#e0a0f0' },
      font: { color: INFERRED_COLOR, size: 11 },
      inferredColor: true,
      title: `inferred — not stated in any document\n${e.relation}` +
             ` (confidence ${(e.confidence || 0).toFixed(2)})\n${e.basis || ''}`,
      inferred: true,
    }));

  for (const e of inferredEdges) tally(counts.inferred, e.kind);
  order.inferred = byCount(counts.inferred);
  inferredLoaded = true;

  // Added hidden and switched off, so nothing about the page changes until a box
  // is ticked. Safe to do at any moment precisely because of the physics flag
  // above; there is no window to wait for.
  if (edgeSet && inferredEdges.length) {
    edgeSet.add(inferredEdges.map(e => Object.assign({}, e, { hidden: true })));
    for (const e of inferredEdges) edgeHidden.set(e.id, true);
  }
  inferredLive = true;

  buildLinkList();               // the group appears; nothing in it is ticked
  applyFilters();
}

// --- the network, built exactly once ----------------------------------------

function build() {
  const container = document.getElementById('net');
  focused = null;
  nodeSet = new vis.DataSet(allNodes);
  edgeSet = new vis.DataSet(allEdges);
  for (const n of allNodes) nodeHidden.set(n.id, false);
  for (const e of allEdges) edgeHidden.set(e.id, false);

  // Filtering used to tear the network down and build a new one, which meant 220
  // fresh stabilisation steps and an entirely new layout for every tick of every
  // box - the graph leapt, and whatever was being read was somewhere else by the
  // time anyone looked up. Ticking six boxes did that six times. Now the network
  // is built once and a filter is nothing but a hidden flag, so the arrangement
  // on screen, and the map of it built in the reader's head, survive filtering.
  //
  // The solver is deliberately left running: an unpinned graph that settles and
  // reacts is the behaviour this page is meant to have.
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
    // The press that dismissed an open dropdown also lands on the canvas, and
    // vis reports it as a click on empty space. One gesture must not both shut
    // the panel and throw away the neighbourhood being read.
    if (swallowCanvasClick) { swallowCanvasClick = false; return; }
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

// --- focus ------------------------------------------------------------------

// Bring one node and everything it touches forward, and push the rest back until
// the next click. Called with null to restore.
function focus(id) {
  if (!nodeSet || !edgeSet) return;
  focused = id;
  // Every edge the DataSet holds, hidden ones included, on BOTH branches. A dim
  // written onto an edge while it was visible and then hidden by a filter would
  // still be on it when another filter brought it back, and it would return as
  // part of a neighbourhood nobody had selected.
  const all = edgeSet.get();

  if (id === null) {
    nodeSet.update(nodeSet.get().map(n => ({
      id: n.id, opacity: 1, font: { color: '#e6e8ec' },
    })));
    edgeSet.update(all.map(e => ({
      id: e.id,
      color: e.inferredColor
        ? { color: INFERRED_COLOR, opacity: 0.75, highlight: '#e0a0f0' }
        : { color: EDGE_COLOR, highlight: '#4a9eff' },
      font: { color: e.inferredColor ? INFERRED_COLOR : '#939aa8' },
    })));
    return;
  }

  // Only what is on screen counts as a neighbour. A hidden inferred edge must
  // not light up an entity the operator has no drawn connection to.
  const near = new Set([id]);
  const touching = new Set();
  for (const e of all) {
    if (e.hidden) continue;
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
  edgeSet.update(all.map(e => ({
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

// --- entity panel -----------------------------------------------------------
// Deliberately not filtered. The panel is the record for that entity, not a view
// of the canvas: hiding a quoted fact because its predicate is unticked would
// make a filter look like an absence of evidence. Said on the panel itself once
// anything is filtered, because that is the moment it could mislead.

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

  // A fixed literal with nothing interpolated into it, so it is safe in this
  // template even though everything around it is escaped by hand.
  const note = isFiltered()
    ? '<p class="panel-note">The graph is filtered; this list is not — it is ' +
      'every connection on the record.</p>'
    : '';

  panel.innerHTML = `
    <h2>${escapeHtml(detail.name)}</h2>
    <p class="small">${escapeHtml(detail.type)} · ${detail.facts.length} connection(s)</p>
    ${note}
    <ul class="facts">${facts || '<li class="panel-empty">no connections</li>'}</ul>`;
}

function isFiltered() {
  return selected.type.size < order.type.length
    || selected.stated.size < order.stated.length
    || selected.inferred.size > 0
    || searchEl.value.trim() !== '';
}

function escapeHtml(value) {
  const div = document.createElement('div');
  div.textContent = value == null ? '' : String(value);
  return div.innerHTML;
}

// --- building the checkbox lists -------------------------------------------
// Every row is built out of nodes and written with textContent. A predicate is
// free text a model produced from a document nobody vetted: it is listed exactly
// as it stands, unreworded and unflagged, in frequency order, and it is never
// handed to the parser as markup.

function makeRow(kind, value, styleSwatch) {
  const row = document.createElement('div');
  row.className = 'dd-row';
  row.dataset.kind = kind;
  row.dataset.value = value;

  const pick = document.createElement('label');
  pick.className = 'dd-pick';
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = selected[kind].has(value);
  const sw = document.createElement('span');
  sw.className = 'dd-sw';
  sw.setAttribute('aria-hidden', 'true');
  styleSwatch(sw, value);
  const text = document.createElement('span');
  text.className = 'dd-label';
  text.textContent = value;
  text.title = value;       // the clamp shows two lines; the title shows all
  pick.append(box, sw, text);

  const count = document.createElement('span');
  count.className = 'dd-count';
  count.textContent = String(counts[kind].get(value) || 0);

  const only = document.createElement('button');
  only.type = 'button';
  only.className = 'dd-only';
  only.textContent = 'only';
  only.title = 'show only this one';
  only.setAttribute('aria-label', 'show only ' + value);
  // Out of the tab order on purpose: 52 rows would otherwise put 104 stops
  // inside one popover. The keyboard route to the same place is written on
  // screen in the popover's footer, not left in a comment here.
  only.tabIndex = -1;

  row.append(pick, count, only);
  return row;
}

function buildGroup(kind, title, note, total, styleSwatch) {
  const group = document.createElement('div');
  group.className = 'dd-group';
  group.dataset.kind = kind;

  const head = document.createElement('div');
  head.className = 'dd-head';
  const headPick = document.createElement('label');
  headPick.className = 'dd-pick';
  const master = document.createElement('input');
  master.type = 'checkbox';
  master.className = 'dd-master';
  master.dataset.kind = kind;
  const headText = document.createElement('span');
  headText.className = 'dd-title';
  headText.textContent = title;
  headPick.append(master, headText);
  const headCount = document.createElement('span');
  headCount.className = 'dd-count';
  headCount.textContent = String(total);
  head.append(headPick, headCount);
  group.append(head);

  if (note) {
    const p = document.createElement('p');
    p.className = 'dd-note';
    p.textContent = note;
    group.append(p);
  }
  delete subHeads[kind];
  if (!order[kind].length) {
    master.disabled = true;
    return group;
  }

  // Thirty-one of the thirty-nine predicates in this corpus occur exactly once,
  // and they are NOT folded away. In case material the assertion made once is
  // disproportionately the one being hunted for - "has no firsthand knowledge
  // relevant to…" is a singleton and may be the most consequential edge here.
  // What they get instead is one tri-state row above them that ticks or unticks
  // all of them at once: the bulk affordance, with nothing hidden.
  const many = order[kind].filter(v => (counts[kind].get(v) || 0) > 1);
  const once = order[kind].filter(v => (counts[kind].get(v) || 0) <= 1);
  const split = once.length >= SPLIT_SINGLETONS_AT && many.length > 0;

  for (const value of (split ? many : order[kind])) {
    group.append(makeRow(kind, value, styleSwatch));
  }
  if (split) {
    const sub = document.createElement('div');
    sub.className = 'dd-row dd-sub';
    const subPick = document.createElement('label');
    subPick.className = 'dd-pick';
    const subBox = document.createElement('input');
    subBox.type = 'checkbox';
    subBox.className = 'dd-subhead';
    subBox.dataset.kind = kind;
    const subText = document.createElement('span');
    subText.className = 'dd-label';
    subText.textContent = 'occurs once';
    subPick.append(subBox, subText);
    const subCount = document.createElement('span');
    subCount.className = 'dd-count';
    subCount.textContent = String(once.length);
    sub.append(subPick, subCount);
    group.append(sub);
    subHeads[kind] = { box: subBox, keys: once };
    for (const value of once) group.append(makeRow(kind, value, styleSwatch));
  }
  return group;
}

function buildTypeList() {
  const list = document.getElementById('typeList');
  list.textContent = '';
  list.append(buildGroup('type', 'Entity types', '', allNodes.length, (sw, value) => {
    sw.classList.add('dot');
    sw.style.background = TYPE_COLORS[value] || OTHER_COLOR;
  }));
  syncBoxes();
}

function buildLinkList() {
  const list = document.getElementById('linkList');
  list.textContent = '';

  list.append(buildGroup('stated', 'Stated in a document',
    `${order.stated.length} kinds. Every one of these came off a page and ` +
    `carries a quote, a file and a page number.`,
    allEdges.length, (sw) => { sw.style.borderTopColor = LIT_EDGE; }));

  const note = !inferredLoaded
    ? 'loading…'
    : !order.inferred.length
      ? 'No pass has been run yet — see the Links page.'
      : `${order.inferred.length} kinds. Not testimony: a model's reading of ` +
        `two entities that were in documents. No quote and no page, which is ` +
        `why they are drawn dashed and why they start switched off.`;
  list.append(buildGroup('inferred', 'Inferred by a model', note,
    inferredEdges.length, (sw) => {
      sw.classList.add('dashed');
      sw.style.borderTopColor = INFERRED_COLOR;
    }));
  syncBoxes();
  filterRows();
}

// --- the popovers -----------------------------------------------------------

let openDd = null;
let swallowCanvasClick = false;

function open(dd) {
  if (openDd === dd) return;
  close();
  openDd = dd;
  dd.classList.add('open');
  dd.querySelector('.dd-pop').hidden = false;
  dd.querySelector('.dd-btn').setAttribute('aria-expanded', 'true');
  const find = dd.querySelector('.dd-find');
  if (find) find.focus();
}

function close(returnFocus) {
  if (!openDd) return;
  const btn = openDd.querySelector('.dd-btn');
  openDd.querySelector('.dd-pop').hidden = true;
  openDd.classList.remove('open');
  btn.setAttribute('aria-expanded', 'false');
  openDd = null;
  if (returnFocus) btn.focus();
}

for (const id of ['typeDd', 'linkDd']) {
  const dd = document.getElementById(id);
  const btn = dd.querySelector('.dd-btn');
  const pop = dd.querySelector('.dd-pop');

  btn.addEventListener('click', () => { openDd === dd ? close(true) : open(dd); });
  btn.addEventListener('keydown', ev => {
    if (ev.key === 'ArrowDown') { ev.preventDefault(); open(dd); }
  });

  // Pressing on the popover's own padding would otherwise blur whatever control
  // has focus, and the focusout below would read that as leaving and shut the
  // panel under the operator's hand. Refusing the default on dead space means
  // focus never moves, so nothing has to be undone.
  pop.addEventListener('mousedown', ev => {
    if (!ev.target.closest('label, input, button')) ev.preventDefault();
  });

  // Fifty-two checkboxes is a long way to Tab. Arrow keys walk the rows that
  // are actually on screen, so the in-popover text filter narrows the walk too.
  pop.addEventListener('keydown', ev => {
    if (ev.key !== 'ArrowDown' && ev.key !== 'ArrowUp') return;
    const stops = [...pop.querySelectorAll('input, button')].filter(el => el.offsetParent);
    const at = stops.indexOf(document.activeElement);
    const next = stops[at + (ev.key === 'ArrowDown' ? 1 : -1)];
    if (next) { next.focus(); ev.preventDefault(); }
  });

  // Tabbing past the last control closes it. Checked a tick later because at
  // focusout time the new element is not focused yet.
  dd.addEventListener('focusout', () => setTimeout(() => {
    if (openDd === dd && !dd.contains(document.activeElement)) close(false);
  }, 0));
}

document.addEventListener('pointerdown', ev => {
  if (openDd && !ev.target.closest('.dd')) {
    // vis will report this same press as a click on empty canvas a moment from
    // now. Mark it so the handler there leaves the focused neighbourhood alone.
    if (ev.target.closest('#net')) swallowCanvasClick = true;
    close(false);
  }
});

// Escape is two things in a fixed order: shut the panel if one is open,
// otherwise release the focused neighbourhood. One handler, so the first press
// never does both at once and leaves the operator wondering which it did.
document.addEventListener('keydown', ev => {
  if (ev.key !== 'Escape') return;
  if (openDd) { close(true); return; }
  if (focused !== null) focus(null);
});

// --- wiring the boxes -------------------------------------------------------

function onListChange(ev) {
  const box = ev.target;
  if (!box.matches('input[type=checkbox]')) return;
  if (box.classList.contains('dd-master')) {
    const kind = box.dataset.kind;
    selected[kind] = box.checked ? new Set(order[kind]) : new Set();
  } else if (box.classList.contains('dd-subhead')) {
    const kind = box.dataset.kind;
    const keys = (subHeads[kind] || { keys: [] }).keys;
    for (const k of keys) {
      if (box.checked) selected[kind].add(k); else selected[kind].delete(k);
    }
  } else {
    const row = box.closest('.dd-row');
    if (!row) return;
    const set = selected[row.dataset.kind];
    if (box.checked) set.add(row.dataset.value); else set.delete(row.dataset.value);
  }
  syncBoxes();
  applyFilters();
}

function onListClick(ev) {
  const only = ev.target.closest('.dd-only');
  if (!only) return;
  const row = only.closest('.dd-row');
  const kind = row.dataset.kind;
  if (kind === 'type') {
    selected.type = new Set([row.dataset.value]);
  } else {
    // "Only this" means only this, across both families - clearing the other
    // one is the whole point of the button.
    selected.stated = new Set();
    selected.inferred = new Set();
    selected[kind].add(row.dataset.value);
  }
  syncBoxes();
  applyFilters();
}

for (const id of ['typeList', 'linkList']) {
  const list = document.getElementById(id);
  list.addEventListener('change', onListChange);
  list.addEventListener('click', onListClick);
}

document.querySelector('[data-reset="type"]').addEventListener('click', () => {
  selected.type = new Set(order.type);
  syncBoxes(); applyFilters();
});
document.querySelector('[data-reset="link"]').addEventListener('click', () => {
  selected.stated = new Set(order.stated);
  selected.inferred = new Set();
  hideOrphans = false;
  document.getElementById('hideOrphans').checked = false;
  syncBoxes(); applyFilters();
});
document.getElementById('hideOrphans').addEventListener('change', ev => {
  hideOrphans = ev.target.checked;
  applyFilters();
});

const linkFind = document.getElementById('linkFind');
linkFind.addEventListener('input', filterRows);

// Hides rows from view. It never touches what is ticked: a master that quietly
// meant "only the ones you can currently see" is how you switch off something
// you were not looking at.
function filterRows() {
  const q = linkFind.value.trim().toLowerCase();
  const list = document.getElementById('linkList');
  let any = false;
  for (const row of list.querySelectorAll('.dd-row')) {
    if (row.classList.contains('dd-sub')) { row.hidden = !!q; continue; }
    const hit = !q || row.dataset.value.toLowerCase().includes(q);
    row.hidden = !hit;
    if (hit) any = true;
  }
  for (const group of list.querySelectorAll('.dd-group')) {
    group.hidden = !!q && !group.querySelector('.dd-row:not(.dd-sub):not([hidden])');
  }
  document.getElementById('linkNone').hidden = !q || any;
}

function syncBoxes() {
  for (const kind of ['type', 'stated', 'inferred']) {
    const master = document.querySelector(`.dd-master[data-kind="${kind}"]`);
    if (master) {
      const n = selected[kind].size, all = order[kind].length;
      master.checked = all > 0 && n === all;
      master.indeterminate = n > 0 && n < all;
    }
    const sub = subHeads[kind];
    if (sub) {
      const on = sub.keys.filter(k => selected[kind].has(k)).length;
      sub.box.checked = on === sub.keys.length && sub.keys.length > 0;
      sub.box.indeterminate = on > 0 && on < sub.keys.length;
    }
    for (const row of document.querySelectorAll(`.dd-row[data-kind="${kind}"]`)) {
      if (row.classList.contains('dd-sub')) continue;
      row.querySelector('input').checked = selected[kind].has(row.dataset.value);
    }
  }
  summarise();
}

// --- the closed state -------------------------------------------------------

function summarise() {
  const t = selected.type.size, T = order.type.length;
  document.getElementById('typeSum').textContent =
    !T ? 'none' :
    t === T ? 'all' :
    t === 0 ? 'none' :
    t <= 2 ? order.type.filter(x => selected.type.has(x)).join(', ') :
    `${t} of ${T}`;

  const e = selected.stated.size, E = order.stated.length;
  const i = selected.inferred.size, I = order.inferred.length;
  // Never one merged count. "42 of 52" would be a number in which a quoted fact
  // and a model's guess weigh exactly the same.
  document.getElementById('linkSum').textContent =
    e === 0 && i === 0 ? 'none' :
    e === E && i === 0 ? 'all stated' :
    e === E && i === I && I > 0 ? 'all + inferred' :
    e === 0 && i === I && I > 0 ? 'inferred only' :
    `${e}/${E} stated · ${i}/${I} inferred`;
  document.getElementById('linkBtn').title =
    `${e} of ${E} stated kinds, ${i} of ${I} inferred kinds`;
}

// --- filtering, without rebuilding anything ---------------------------------

function applyFilters() {
  if (!nodeSet) return;
  const term = searchEl.value.trim().toLowerCase();
  const searching = term.length > 0;

  const candidates = new Set();
  for (const n of allNodes) {
    if (!selected.type.has(n.entityType)) continue;
    if (searching && !n.label.toLowerCase().includes(term)) continue;
    candidates.add(n.id);
  }

  const linked = new Set();
  const edgeUpdates = [];
  let statedShown = 0, inferredShown = 0;

  const sweep = (list, isInferred) => {
    const picked = isInferred ? selected.inferred : selected.stated;
    for (const e of list) {
      const on = picked.has(e.kind) && candidates.has(e.from) && candidates.has(e.to);
      if (on) {
        linked.add(e.from); linked.add(e.to);
        if (isInferred) inferredShown++; else statedShown++;
      }
      if (edgeHidden.get(e.id) !== !on) {
        edgeHidden.set(e.id, !on);
        edgeUpdates.push({ id: e.id, hidden: !on });
      }
    }
  };
  sweep(allEdges, false);
  // Only once they are actually in the DataSet. update() on an id vis has never
  // seen does not fail, it inserts - an edge with a hidden flag and no ends.
  if (inferredLive) sweep(inferredEdges, true);

  let orphans = 0;
  const nodeUpdates = [];
  for (const n of allNodes) {
    let on = candidates.has(n.id);
    // The unconnected rule is suspended whenever the search box has text. You
    // typed a name; the answer to that is the name, connected or not. Without
    // this, searching an entity whose only link is unticked makes it vanish and
    // reads as "not in this corpus", which is the worst thing a filter can say.
    if (on && !searching && !linked.has(n.id)) {
      orphans++;
      if (hideOrphans) on = false;
    }
    if (nodeHidden.get(n.id) !== !on) {
      nodeHidden.set(n.id, !on);
      nodeUpdates.push({ id: n.id, hidden: !on });
    }
  }

  if (nodeUpdates.length) nodeSet.update(nodeUpdates);
  if (edgeUpdates.length) edgeSet.update(edgeUpdates);

  // Re-applied rather than thrown away. "Click the person, then pare the links
  // back around them" is the whole reason to have a link filter, and clearing
  // the focus on every tick made it impossible. The neighbourhood is recomputed
  // because the filter may have changed which edges count as touching - and if
  // the focused entity is no longer on screen the focus is released, because a
  // dimmed graph around an invisible centre is just a dimmed graph.
  if (focused !== null) {
    const still = nodeSet.get(focused);
    focus(still && !still.hidden ? focused : null);
  }

  const cut = (hideOrphans && !searching) ? orphans : 0;
  writeStatus(candidates.size - cut, cut, statedShown, inferredShown);
  writeOrphanLabel(orphans, searching);
  markInferenceOnly(statedShown, inferredShown, candidates.size - cut);
}

// Nothing on screen came out of a document, and the only place that can be said
// where it will survive a screenshot is inside the canvas box itself.
function markInferenceOnly(statedShown, inferredShown, shown) {
  const only = shown > 0 && inferredShown > 0 && statedShown === 0;
  wrap.classList.toggle('inference-only', only);
  banner.hidden = !only;
}

function writeStatus(shown, orphansCut, statedShown, inferredShown) {
  if (!shown) {
    statusEl.textContent =
      `nothing matches these filters — all ${allNodes.length} entities are hidden`;
    return;
  }
  const parts = [];
  parts.push(shown === allNodes.length
    ? `${allNodes.length} entities`
    : `${shown} of ${allNodes.length} entities`);
  // A word, not a zero. A zero in a run of numbers is skimmed past; "no stated
  // links" is the one clause on this line that has to be read.
  parts.push(!statedShown ? 'no stated links'
    : statedShown === allEdges.length ? `${statedShown} stated links`
    : `${statedShown} of ${allEdges.length} stated links`);
  if (inferredLoaded) {
    parts.push(!inferredEdges.length ? 'no inferred links yet'
      : !selected.inferred.size ? 'inferred off'
      : inferredShown === inferredEdges.length ? `${inferredShown} inferred links`
      : `${inferredShown} of ${inferredEdges.length} inferred`);
  }
  if (orphansCut) parts.push(`${orphansCut} with no links hidden`);
  statusEl.textContent = parts.join(' · ');
}

function writeOrphanLabel(orphans, searching) {
  const box = document.getElementById('hideOrphans');
  const text = document.getElementById('hideOrphansText');
  if (searching) {
    box.disabled = true;
    text.textContent = 'not while you are searching — a name you look for is always shown';
    return;
  }
  box.disabled = !orphans && !hideOrphans;
  const word = orphans === 1 ? 'entity' : 'entities';
  text.textContent = !orphans ? 'nothing is left without a link'
    : hideOrphans ? `hiding ${orphans} ${word} with no links left`
    : `hide ${orphans} ${word} left with no links`;
}

// --- the rest of the toolbar ------------------------------------------------

searchEl.addEventListener('input', debounce(applyFilters, 250));

document.getElementById('refit').addEventListener('click', () => {
  if (!network) return;
  // Fit what is on screen. Fitting everything would frame 128 entities when 12
  // are drawn, and the filtered graph would be a speck in the middle of it.
  const ids = nodeSet.get({ filter: n => !n.hidden }).map(n => n.id);
  network.fit(ids.length ? { nodes: ids, animation: true } : { animation: true });
});

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

load();
