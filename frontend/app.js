const form = document.getElementById('compare-form');
const button = document.getElementById('analyze-button');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const cardsEl = document.getElementById('issue-cards');
const timelineEl = document.getElementById('timeline');
const traceEl = document.getElementById('agent-trace');

for (const [inputId, labelId] of [['video-a','video-a-name'],['video-b','video-b-name']]) {
  document.getElementById(inputId).addEventListener('change', (event) => {
    const file = event.target.files?.[0];
    document.getElementById(labelId).textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB` : 'Choose video';
  });
}

function formatTime(seconds) {
  const value = Number(seconds || 0);
  const mins = Math.floor(value / 60);
  const secs = (value % 60).toFixed(2).padStart(5, '0');
  return `${mins}:${secs}`;
}

function renderResults(payload) {
  const analysis = payload.evidence?.[0] || {};
  const cards = analysis.evidence_cards || [];
  const timeline = analysis.timeline || [];
  const trace = analysis.agent_trace || [];
  const action = analysis.disposition || 'UNKNOWN';
  const verdict = analysis.final_verdict || {status: action, summary: ''};
  const comparative = analysis.comparative_analysis || {};
  const alignment = comparative.alignment || {};

  const verdictEl = document.getElementById('final-verdict');
  verdictEl.textContent = verdict.status || 'UNKNOWN';
  verdictEl.className = `disposition ${(verdict.status || 'unknown').toLowerCase()}`;

  document.getElementById('disposition').textContent = action;
  document.getElementById('disposition').className = `disposition ${action.toLowerCase()}`;
  document.getElementById('comparative-count').textContent = analysis.comparative_issue_count ?? 0;
  document.getElementById('confirmed-count').textContent = analysis.total_confirmed_issues ?? 0;
  document.getElementById('issue-count').textContent = analysis.total_issues ?? cards.length;
  document.getElementById('raw-event-count').textContent = analysis.total_raw_events ?? analysis.total_issues ?? cards.length;
  document.getElementById('job-id').textContent = payload.job_id || '—';
  document.getElementById('verdict-summary').textContent = verdict.summary || 'No summary available.';

  if (comparative.summary) {
    const score = Number(alignment.score || 0);
    document.getElementById('alignment-summary').textContent = `${comparative.summary} Alignment score ${(score * 100).toFixed(1)}%.`;
  } else {
    document.getElementById('alignment-summary').textContent = 'Pairwise alignment data unavailable.';
  }

  traceEl.innerHTML = '';
  if (!trace.length) {
    traceEl.innerHTML = '<p class="empty">No second-pass or pairwise actions were required.</p>';
  } else {
    for (const step of trace) {
      const node = document.createElement('div');
      node.className = 'timeline-item';
      node.textContent = `Video ${step.video} · ${step.type.replaceAll('_',' ')} · ${formatTime(step.start_time)}–${formatTime(step.end_time)} · ${step.result.toUpperCase()} — ${step.reason}`;
      traceEl.appendChild(node);
    }
  }

  timelineEl.innerHTML = '';
  if (!timeline.length) {
    timelineEl.innerHTML = '<p class="empty">No reviewable visual anomalies detected.</p>';
  } else {
    for (const item of timeline) {
      const node = document.createElement('div');
      node.className = 'timeline-item';
      node.textContent = `Video ${item.video} · ${item.type.replaceAll('_',' ')} · ${formatTime(item.start_time)}–${formatTime(item.end_time)} · ${item.severity}`;
      timelineEl.appendChild(node);
    }
  }

  cardsEl.innerHTML = '';
  if (!cards.length) {
    cardsEl.innerHTML = '<p class="empty">No representative evidence cards were generated.</p>';
  } else {
    for (const card of cards) {
      const article = document.createElement('article');
      article.className = 'issue-card';
      const image = card.evidence_frames?.[0];
      const rawCount = card.details?.raw_event_count || 1;
      article.innerHTML = `
        ${image ? `<img src="${image}" alt="Evidence for ${card.issue_id}" loading="lazy" />` : ''}
        <div class="issue-body">
          <div class="issue-top">
            <h3>${card.issue_id} · ${card.type.replaceAll('_',' ')}</h3>
            <span class="badge">${card.severity}</span>
          </div>
          <p class="issue-meta">
            Video ${card.video} · ${formatTime(card.start_time)}–${formatTime(card.end_time)}<br />
            Confidence ${(Number(card.confidence || 0) * 100).toFixed(1)}% · ${rawCount} raw event${rawCount === 1 ? '' : 's'} grouped
          </p>
        </div>`;
      cardsEl.appendChild(article);
    }
  }

  resultsEl.classList.remove('hidden');
  resultsEl.scrollIntoView({behavior:'smooth', block:'start'});
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const a = document.getElementById('video-a').files?.[0];
  const b = document.getElementById('video-b').files?.[0];
  if (!a || !b) return;

  const data = new FormData();
  data.append('video_a', a);
  data.append('video_b', b);

  button.disabled = true;
  button.textContent = 'Analyzing…';
  statusEl.textContent = 'Aligning A/B, comparing matched frames, running OpenCV QA, and applying targeted confirmation.';
  resultsEl.classList.add('hidden');

  try {
    const response = await fetch('/compare', {method:'POST', body:data});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Analysis failed');
    renderResults(payload);
    statusEl.textContent = 'Analysis complete.';
  } catch (error) {
    statusEl.textContent = `Error: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = 'Analyze videos';
  }
});
