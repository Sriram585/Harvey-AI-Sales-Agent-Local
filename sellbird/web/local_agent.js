/**
 * Harvey AI / Sellbird Local Agent UI
 * 100% Offline client-side logic. Zero remote network dependencies.
 */

(function () {
  'use strict';

  let currentCompany = null;
  let currentProspectId = null;
  let allProspects = [];

  // =========================================================================
  // API Client with 503 Ollama Failure Detection
  // =========================================================================
  async function api(path, options = {}) {
    const res = await fetch(path, {
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
      ...options,
    });

    const data = await res.json().catch(() => ({}));

    // Detect Ollama offline 503 failure
    if (res.status === 503 && data.ollama_offline) {
      showOfflineBanner(data.message, data.queued_task_id);
      return { _error: true, status: 503, ...data };
    }

    if (!res.ok) {
      const err = new Error(data.message || data.detail || `HTTP ${res.status}`);
      err.data = data;
      err.status = res.status;
      throw err;
    }

    return data;
  }

  function showOfflineBanner(message, taskId) {
    const banner = document.getElementById('offline-banner');
    const msgEl = document.getElementById('banner-msg');
    if (banner && msgEl) {
      let text = message || 'Ollama daemon is unreachable on http://127.0.0.1:11434.';
      if (taskId) {
        text += ` (AI task queued as Job #${taskId} for processing upon recovery).`;
      }
      msgEl.textContent = text;
      banner.classList.remove('hidden');
    }
  }

  function hideOfflineBanner() {
    const banner = document.getElementById('offline-banner');
    if (banner) banner.classList.add('hidden');
  }

  // =========================================================================
  // System Status & Top Bar
  // =========================================================================
  async function fetchSystemStatus() {
    try {
      const st = await api('/api/local/system/status');

      // Hardware Profile
      const hw = st.hardware_profile || {};
      const hwText = `${(hw.tier || 'laptop').toUpperCase()} (${hw.cpu_cores || 0} Cores, ${(hw.ram_gb || 0).toFixed(1)} GB RAM)`;
      document.getElementById('hw-tier').textContent = hwText;
      document.getElementById('hw-chip').title = hw.rationale || 'Hardware profile';

      // Ollama Health
      const healthEl = document.getElementById('ollama-health');
      if (st.ollama_online) {
        healthEl.textContent = 'LIVE';
        healthEl.className = 'health-badge status-live';
        hideOfflineBanner();
      } else {
        healthEl.textContent = 'OFFLINE';
        healthEl.className = 'health-badge status-offline';
        showOfflineBanner();
      }

      // Model Selector
      const modelSelect = document.getElementById('model-select');
      const models = st.installed_models && st.installed_models.length > 0
        ? st.installed_models
        : [st.active_model || 'qwen3:4b'];

      modelSelect.innerHTML = '';
      models.forEach((m) => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        if (m === st.active_model) opt.selected = true;
        modelSelect.appendChild(opt);
      });

      // Strict Offline Toggle
      document.getElementById('offline-toggle').checked = !!st.strict_offline_mode;

      // Pending Queue Count
      document.getElementById('queue-count').textContent = st.pending_queue_count || 0;
    } catch (e) {
      console.error('Failed to fetch system status:', e);
    }
  }

  // =========================================================================
  // Navigation Tabs
  // =========================================================================
  function initTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        tabs.forEach((t) => t.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach((p) => p.classList.remove('active'));

        tab.classList.add('active');
        const targetId = tab.dataset.tab;
        const targetPane = document.getElementById(targetId);
        if (targetPane) targetPane.classList.add('active');

        // Load data on tab switch
        if (targetId === 'tab-company') loadCompanyAndProducts();
        if (targetId === 'tab-prospects') loadProspects();
        if (targetId === 'tab-replies') loadSampleReplies();
        if (targetId === 'tab-telemetry') loadTelemetryAndQueue();
      });
    });
  }

  // =========================================================================
  // Tab 1: Company & Products
  // =========================================================================
  async function loadCompanyAndProducts() {
    try {
      const companies = await api('/api/local/companies');
      if (companies && companies.length > 0) {
        currentCompany = companies[0];
        document.getElementById('comp-name').value = currentCompany.name || '';
        document.getElementById('comp-website').value = currentCompany.website || '';
        document.getElementById('comp-notes').value = currentCompany.raw_notes || '';

        const profEl = document.getElementById('comp-profile-json');
        try {
          const prof = JSON.parse(currentCompany.profile_json || '{}');
          profEl.textContent = JSON.stringify(prof, null, 2);
        } catch {
          profEl.textContent = currentCompany.profile_json || 'No profile generated yet.';
        }

        // Load Products
        const products = await api(`/api/local/products?company_id=${currentCompany.id}`);
        renderProducts(products);
      }
    } catch (e) {
      console.error('Error loading company data:', e);
    }
  }

  function renderProducts(products) {
    const container = document.getElementById('products-list');
    document.getElementById('product-count').textContent = products.length;

    if (!products || products.length === 0) {
      container.innerHTML = '<p class="empty-state">No products registered yet.</p>';
      return;
    }

    container.innerHTML = '';
    products.forEach((p) => {
      const card = document.createElement('div');
      card.className = 'product-card';
      card.innerHTML = `
        <div class="product-card-top">
          <span class="product-name">${escapeHtml(p.name)}</span>
          <span class="product-tier">${escapeHtml(p.pricing_tier || 'Standard')}</span>
        </div>
        <p class="product-desc">${escapeHtml(p.description || '')}</p>
        <p class="product-vp"><strong>Value Prop:</strong> ${escapeHtml(p.value_prop || '')}</p>
        <div class="flex-between" style="margin-top: 8px;">
          <button class="btn-secondary btn-sm btn-gen-icp" data-id="${p.id}">Generate ICP (with Embedding)</button>
          <span class="chip-label" id="icp-status-${p.id}">Local ICP Active</span>
        </div>
        <div class="icp-preview-box" id="icp-box-${p.id}">Loading ICP details...</div>
      `;
      container.appendChild(card);

      // Fetch existing ICP for preview
      loadProductICP(p.id);
    });

    // Wire ICP generation buttons
    container.querySelectorAll('.btn-gen-icp').forEach((btn) => {
      btn.addEventListener('click', async (e) => {
        const prodId = e.target.dataset.id;
        e.target.disabled = true;
        e.target.textContent = 'Generating ICP...';
        try {
          const res = await api(`/api/local/products/${prodId}/icp`, { method: 'POST' });
          if (res.status === 'success' || res.deterministic_fallback) {
            await loadProductICP(prodId);
          }
        } catch (err) {
          alert(`ICP Generation failed: ${err.message}`);
        } finally {
          e.target.disabled = false;
          e.target.textContent = 'Generate ICP (with Embedding)';
          fetchSystemStatus();
        }
      });
    });
  }

  async function loadProductICP(productId) {
    try {
      const icps = await api('/api/local/icps');
      const prodIcp = icps.filter((i) => i.product_id === parseInt(productId)).pop();
      const box = document.getElementById(`icp-box-${productId}`);
      if (box) {
        if (prodIcp) {
          const parsed = JSON.parse(prodIcp.icp_json || '{}');
          box.innerHTML = `
            <strong>Target Roles:</strong> ${(parsed.target_roles || []).join(', ') || 'N/A'}<br>
            <strong>Pain Points:</strong> ${(parsed.pain_points || []).join('; ') || 'N/A'}<br>
            <strong>Triggers:</strong> ${(parsed.buying_triggers || []).join('; ') || 'N/A'}
          `;
        } else {
          box.innerHTML = '<em>No ICP generated yet. Click "Generate ICP" above.</em>';
        }
      }
    } catch (e) {
      console.error('Error loading ICP:', e);
    }
  }

  // =========================================================================
  // Tab 2: Prospects & Multi-Product Fit Matrix
  // =========================================================================
  async function loadProspects() {
    try {
      allProspects = await api('/api/local/prospects');
      renderProspectsTable(allProspects);
    } catch (e) {
      console.error('Error loading prospects:', e);
    }
  }

  async function renderProspectsTable(prospects) {
    const tbody = document.getElementById('prospects-tbody');
    if (!prospects || prospects.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="loading-td">No prospects found.</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    for (const p of prospects) {
      const scores = p.scores || [];
      const latestScore = scores[0] || {};
      const priority = latestScore.priority || 'Medium';
      const totalScore = latestScore.total_score != null ? latestScore.total_score : '--';

      const fit = p.product_fit_matrix || {};
      const primaryProdName = fit.primary_product_id
        ? (fit.primary_product_id === 1 ? 'CloudCost Guard' : fit.primary_product_id === 2 ? 'PipeGuard' : 'ShiftSync')
        : 'Auto-detecting...';

      const crossSells = [];
      try {
        const csIds = JSON.parse(fit.cross_sell_product_ids_json || '[]');
        csIds.forEach((id) => {
          if (id === 1) crossSells.push('CloudCost Guard');
          if (id === 2) crossSells.push('PipeGuard');
          if (id === 3) crossSells.push('ShiftSync');
        });
      } catch {}

      const tr = document.createElement('tr');
      tr.id = `prospect-row-${p.id}`;
      tr.innerHTML = `
        <td><strong>${escapeHtml(p.company_name)}</strong></td>
        <td>${escapeHtml(p.contact_name || '')}<br><small style="color:#777;">${escapeHtml(p.contact_role || '')}</small></td>
        <td><span class="badge-priority priority-${priority.toLowerCase()}">${priority}</span></td>
        <td><strong>${totalScore}</strong></td>
        <td><span class="badge-tag" style="background:#000; color:#fff;">${primaryProdName}</span></td>
        <td>${crossSells.map((c) => `<span class="badge-tag">${c}</span>`).join(' ') || '<em style="color:#999;">None</em>'}</td>
        <td><button class="btn-secondary btn-sm btn-inspect" data-id="${p.id}">Inspect &amp; Pipeline</button></td>
      `;
      tbody.appendChild(tr);
    }

    // Wire Inspect buttons
    tbody.querySelectorAll('.btn-inspect').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const id = parseInt(e.target.dataset.id);
        selectProspect(id);
      });
    });
  }

  async function selectProspect(prospectId) {
    currentProspectId = prospectId;

    // Highlight row
    document.querySelectorAll('#prospects-tbody tr').forEach((r) => r.classList.remove('selected'));
    const row = document.getElementById(`prospect-row-${prospectId}`);
    if (row) row.classList.add('selected');

    document.getElementById('btn-run-full-pipeline').disabled = false;

    // Load full details
    try {
      const p = await api(`/api/local/prospects/${prospectId}`);
      document.getElementById('drawer-prospect-name').textContent = p.company_name;
      document.getElementById('drawer-prospect-contact').textContent = `${p.contact_name || ''} — ${p.contact_role || ''} (${p.contact_email || ''})`;
      document.getElementById('drawer-raw-text').textContent = p.raw_text || 'No raw text available.';

      // Research
      const resBox = document.getElementById('drawer-research');
      try {
        const rObj = JSON.parse(p.research_json || '{}');
        if (rObj.summary) {
          resBox.innerHTML = `
            <p><strong>Summary:</strong> ${escapeHtml(rObj.summary)}</p>
            <p style="margin-top:6px;"><strong>Tech Stack:</strong> ${(rObj.tech_stack || []).map((t) => `<span class="badge-tag">${t}</span>`).join(' ')}</p>
          `;
        } else {
          resBox.textContent = 'No research generated yet.';
        }
      } catch {
        resBox.textContent = p.research_json || 'No research generated yet.';
      }

      // Product Fit Matrix
      const fitBox = document.getElementById('drawer-fit-matrix');
      const fit = p.product_fit_matrix;
      if (fit && fit.scores_json) {
        try {
          const scores = JSON.parse(fit.scores_json);
          let html = '<div style="display:flex; flex-direction:column; gap:6px;">';
          for (const [prodName, sim] of Object.entries(scores)) {
            const pct = Math.round(sim * 100);
            html += `
              <div>
                <div class="flex-between">
                  <span><strong>${escapeHtml(prodName)}</strong></span>
                  <span style="font-family:var(--font-mono);">${pct}% Fit (${sim})</span>
                </div>
                <div style="background:#E0E0E0; height:6px; border-radius:2px; margin-top:2px;">
                  <div style="background:#000000; width:${pct}%; height:100%; border-radius:2px;"></div>
                </div>
              </div>
            `;
          }
          html += '</div>';
          fitBox.innerHTML = html;
        } catch {
          fitBox.textContent = 'Error parsing fit scores.';
        }
      } else {
        fitBox.textContent = 'Fit matrix not calculated yet.';
      }

      // Buying Signals
      const sigBox = document.getElementById('drawer-signals');
      const signals = p.buying_signals || [];
      if (signals.length > 0) {
        sigBox.innerHTML = signals.map((s) => `
          <div style="border-bottom:1px solid #EAEAEA; padding:6px 0;">
            <div class="flex-between">
              <span class="badge-tag"><strong>${escapeHtml(s.signal_type.toUpperCase())}</strong></span>
              <span class="${s.is_grounded ? 'badge-grounded-true' : 'badge-grounded-false'}">
                ${s.is_grounded ? '[VERIFIED]' : '[UNVERIFIED]'}
              </span>
            </div>
            <p style="font-size:11px; margin-top:4px;"><em>"${escapeHtml(s.evidence_quote || '')}"</em></p>
            <p style="font-size:11px; color:#666; margin-top:2px;">Confidence: ${(s.confidence * 100).toFixed(0)}% — ${escapeHtml(s.reasoning || '')}</p>
          </div>
        `).join('');
      } else {
        sigBox.textContent = 'No signals classified yet.';
      }

      // Problem Statement
      const probBox = document.getElementById('drawer-problem');
      const problems = p.problem_statements || [];
      if (problems.length > 0) {
        probBox.innerHTML = `<p>${escapeHtml(problems[0].statement)}</p>`;
      } else {
        probBox.textContent = 'No problem statement generated yet.';
      }

      // Score
      const scoreBox = document.getElementById('drawer-score');
      const scores = p.scores || [];
      if (scores.length > 0) {
        const s = scores[0];
        scoreBox.innerHTML = `
          <div class="flex-between" style="margin-bottom:6px;">
            <span class="badge-priority priority-${s.priority.toLowerCase()}">${s.priority} PRIORITY</span>
            <span style="font-size:16px; font-weight:700;">TOTAL: ${s.total_score} / 100</span>
          </div>
          <p style="font-size:11px; color:#555;">Deterministic Fit &amp; Signals: <strong>${s.deterministic_score}</strong> | LLM Strategic: <strong>${s.llm_score}</strong></p>
          <p style="font-size:11px; margin-top:4px; font-style:italic;">${escapeHtml(s.reasoning || '')}</p>
        `;
      } else {
        scoreBox.textContent = 'Not scored yet.';
      }

      // Outreach Email
      const emailBox = document.getElementById('drawer-email');
      const emails = p.outreach_emails || [];
      if (emails.length > 0) {
        const em = emails[0];
        emailBox.innerHTML = `
          <p><strong>Subject:</strong> ${escapeHtml(em.subject)}</p>
          <div style="background:#FFF; border:1px solid #DDD; padding:10px; margin-top:8px; white-space:pre-wrap; font-size:12px;">${escapeHtml(em.body)}</div>
        `;
      } else {
        emailBox.textContent = 'No email drafted yet.';
      }

    } catch (e) {
      console.error('Error selecting prospect:', e);
    }
  }

  // =========================================================================
  // Pipeline Action Buttons
  // =========================================================================
  function initPipelineButtons() {
    // 1. Research
    document.getElementById('btn-pipe-research').addEventListener('click', async () => {
      if (!currentProspectId) return;
      await runPipeStep('Researching...', `/api/local/prospects/${currentProspectId}/research`);
    });

    // 2. Fit Matrix
    document.getElementById('btn-pipe-fit').addEventListener('click', async () => {
      if (!currentProspectId) return;
      await runPipeStep('Calculating Multi-Product Fit...', `/api/local/prospects/${currentProspectId}/product-fit`);
    });

    // 3. Buying Signals
    document.getElementById('btn-pipe-signals').addEventListener('click', async () => {
      if (!currentProspectId) return;
      await runPipeStep('Extracting Buying Signals...', `/api/local/prospects/${currentProspectId}/signals`);
    });

    // 4. Problem Statement
    document.getElementById('btn-pipe-problem').addEventListener('click', async () => {
      if (!currentProspectId) return;
      await runPipeStep('Synthesizing Problem Statement...', `/api/local/prospects/${currentProspectId}/problem-statement`, { json: {} });
    });

    // 5. Score
    document.getElementById('btn-pipe-score').addEventListener('click', async () => {
      if (!currentProspectId) return;
      await runPipeStep('Computing Hybrid Score...', `/api/local/prospects/${currentProspectId}/score`, { json: {} });
    });

    // 6. Email
    document.getElementById('btn-pipe-email').addEventListener('click', async () => {
      if (!currentProspectId) return;
      await runPipeStep('Drafting Cold Email...', `/api/local/prospects/${currentProspectId}/email`, { json: {} });
    });

    // Run Full Pipeline
    document.getElementById('btn-run-full-pipeline').addEventListener('click', async () => {
      if (!currentProspectId) return;
      const btn = document.getElementById('btn-run-full-pipeline');
      btn.disabled = true;
      btn.textContent = 'Running Full Pipeline...';

      const steps = [
        `/api/local/prospects/${currentProspectId}/research`,
        `/api/local/prospects/${currentProspectId}/product-fit`,
        `/api/local/prospects/${currentProspectId}/signals`,
        `/api/local/prospects/${currentProspectId}/problem-statement`,
        `/api/local/prospects/${currentProspectId}/score`,
        `/api/local/prospects/${currentProspectId}/email`,
      ];

      for (const step of steps) {
        try {
          await api(step, { method: 'POST', body: JSON.stringify({}) });
        } catch (e) {
          console.warn('Pipeline step warning:', e);
        }
      }

      await selectProspect(currentProspectId);
      await loadProspects();
      fetchSystemStatus();
      btn.disabled = false;
      btn.textContent = 'Run Full Pipeline';
    });

    // Scraper Bar
    document.getElementById('btn-scrape').addEventListener('click', async () => {
      const urlInput = document.getElementById('scrape-url');
      const compInput = document.getElementById('scrape-company');
      const alertEl = document.getElementById('scrape-alert');
      const url = urlInput.value.trim();

      if (!url) {
        alert('Please enter a valid URL to scrape.');
        return;
      }

      const btn = document.getElementById('btn-scrape');
      btn.disabled = true;
      btn.textContent = 'Scraping...';
      alertEl.classList.add('hidden');

      try {
        const res = await api('/api/local/prospects/scrape', {
          method: 'POST',
          body: JSON.stringify({
            url,
            company_name: compInput.value.trim() || undefined,
          }),
        });

        if (res.status === 'success') {
          alertEl.textContent = `Scraped ${res.company_name} successfully (${res.text_length} chars). Added to SQLite prospects.`;
          alertEl.classList.remove('hidden');
          urlInput.value = '';
          compInput.value = '';
          await loadProspects();
        }
      } catch (err) {
        alertEl.textContent = err.data && err.data.error === 'offline_mode_error'
          ? 'Live web scraping is disabled while STRICT OFFLINE MODE is on. Toggle offline mode off in top bar to scrape.'
          : `Scraping error: ${err.message}`;
        alertEl.classList.remove('hidden');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Scrape to SQLite';
      }
    });
  }

  async function runPipeStep(loadingText, path, opts = {}) {
    try {
      await api(path, { method: 'POST', body: JSON.stringify(opts.json || {}) });
      await selectProspect(currentProspectId);
      await loadProspects();
      fetchSystemStatus();
    } catch (e) {
      alert(`Workflow error: ${e.message}`);
    }
  }

  // =========================================================================
  // Tab 3: Reply Classifier Sandbox
  // =========================================================================
  async function loadSampleReplies() {
    try {
      const replies = await api('/api/local/replies');
      const select = document.getElementById('sample-reply-select');
      select.innerHTML = '<option value="">-- Choose Seeded Sample Reply --</option>';

      replies.forEach((r, idx) => {
        const opt = document.createElement('option');
        opt.value = r.reply_text;
        opt.textContent = `[${r.intent.toUpperCase()}] "${r.reply_text.slice(0, 50)}..."`;
        select.appendChild(opt);
      });

      select.onchange = () => {
        if (select.value) {
          document.getElementById('reply-input-text').value = select.value;
        }
      };
    } catch (e) {
      console.error('Error loading sample replies:', e);
    }
  }

  function initReplySandbox() {
    document.getElementById('btn-classify-reply').addEventListener('click', async () => {
      const text = document.getElementById('reply-input-text').value.trim();
      if (!text) {
        alert('Please enter or select a reply text to classify.');
        return;
      }

      const btn = document.getElementById('btn-classify-reply');
      btn.disabled = true;
      btn.textContent = 'Classifying...';

      try {
        const res = await api('/api/local/replies/classify', {
          method: 'POST',
          body: JSON.stringify({ reply_text: text }),
        });

        const card = document.getElementById('reply-output-card');
        const item = res.deterministic_fallback || res;
        card.className = 'reply-card-result';
        card.innerHTML = `
          <div class="reply-intent-badge">${escapeHtml(item.intent || 'UNKNOWN')}</div>
          <p><strong>Confidence:</strong> ${((item.confidence || 0) * 100).toFixed(1)}%</p>
          <p style="margin-top:6px;"><strong>Recommended Next Action:</strong> <span class="badge-tag" style="background:#000;color:#FFF;">${escapeHtml(item.next_action || 'review')}</span></p>
          <p style="margin-top:8px; font-size:12px; color:#444;"><strong>Reasoning:</strong> ${escapeHtml(item.reasoning || '')}</p>
        `;
      } catch (err) {
        alert(`Classification error: ${err.message}`);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Classify Reply Intent';
        fetchSystemStatus();
      }
    });
  }

  // =========================================================================
  // Tab 4: Telemetry & Task Queue
  // =========================================================================
  async function loadTelemetryAndQueue() {
    // 1. Logs
    try {
      const logs = await api('/api/local/logs/inference?limit=50');
      const tbody = document.getElementById('logs-tbody');
      if (!logs || logs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="loading-td">No inference logs yet.</td></tr>';
      } else {
        tbody.innerHTML = logs.map((l) => `
          <tr>
            <td style="font-family:var(--font-mono);">${escapeHtml(l.timestamp ? l.timestamp.slice(11, 19) : '')}</td>
            <td><strong>${escapeHtml(l.workflow_name || '')}</strong></td>
            <td style="font-family:var(--font-mono); font-size:11px;">${escapeHtml(l.endpoint_url || '')}</td>
            <td style="font-family:var(--font-mono);">${escapeHtml(l.model_name || '')}</td>
            <td>${l.latency_ms ? l.latency_ms.toFixed(0) + ' ms' : '0'}</td>
            <td>${l.eval_count || 0}</td>
            <td>${l.tokens_per_sec ? l.tokens_per_sec.toFixed(1) : 0}</td>
            <td>${l.json_repaired ? 'Yes' : 'No'}</td>
            <td><span class="badge-tag" style="${l.status === 'success' ? 'background:#000;color:#FFF;' : 'background:#FFF;border:1px solid #000;'}">${l.status}</span></td>
          </tr>
        `).join('');
      }
    } catch (e) {
      console.error('Error loading logs:', e);
    }

    // 2. Queue
    try {
      const tasks = await api('/api/local/queue?limit=50');
      const qbody = document.getElementById('queue-tbody');
      if (!tasks || tasks.length === 0) {
        qbody.innerHTML = '<tr><td colspan="7" class="loading-td">No tasks in queue.</td></tr>';
      } else {
        qbody.innerHTML = tasks.map((t) => `
          <tr>
            <td style="font-family:var(--font-mono);">${t.id}</td>
            <td><strong>${escapeHtml(t.task_type)}</strong></td>
            <td><span class="badge-tag" style="${t.status === 'completed' ? 'background:#000;color:#FFF;' : ''}">${t.status}</span></td>
            <td style="font-family:var(--font-mono); font-size:11px; max-width:200px; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(t.payload_json || '{}')}</td>
            <td style="font-size:11px; color:#C00;">${escapeHtml(t.error_msg || '')}</td>
            <td style="font-family:var(--font-mono);">${escapeHtml(t.created_at ? t.created_at.slice(11, 19) : '')}</td>
            <td style="font-family:var(--font-mono);">${escapeHtml(t.completed_at ? t.completed_at.slice(11, 19) : '--')}</td>
          </tr>
        `).join('');
      }
    } catch (e) {
      console.error('Error loading queue:', e);
    }
  }

  // =========================================================================
  // Global Event Wiring
  // =========================================================================
  function initControls() {
    // Model Select
    document.getElementById('model-select').addEventListener('change', async (e) => {
      try {
        await api('/api/local/system/config', {
          method: 'POST',
          body: JSON.stringify({ active_model: e.target.value }),
        });
        fetchSystemStatus();
      } catch (err) {
        alert(`Model switch failed: ${err.message}`);
      }
    });

    // Offline Toggle
    document.getElementById('offline-toggle').addEventListener('change', async (e) => {
      try {
        await api('/api/local/system/config', {
          method: 'POST',
          body: JSON.stringify({ strict_offline_mode: e.target.checked }),
        });
        fetchSystemStatus();
      } catch (err) {
        alert(`Failed to toggle offline mode: ${err.message}`);
      }
    });

    // Process Queue Button
    document.getElementById('btn-process-queue').addEventListener('click', async () => {
      const btn = document.getElementById('btn-process-queue');
      btn.disabled = true;
      btn.textContent = 'Draining...';
      try {
        const res = await api('/api/local/queue/process', { method: 'POST' });
        alert(`Queue processing complete: ${res.processed} processed, ${res.failed} failed.`);
        await loadTelemetryAndQueue();
        fetchSystemStatus();
      } catch (err) {
        alert(`Queue process error: ${err.message}`);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Process Queue';
      }
    });

    // Banner Retry Button
    document.getElementById('btn-banner-retry').addEventListener('click', () => {
      fetchSystemStatus();
    });

    // Generate Company Profile
    document.getElementById('btn-gen-profile').addEventListener('click', async () => {
      if (!currentCompany) return;
      const btn = document.getElementById('btn-gen-profile');
      btn.disabled = true;
      btn.textContent = 'Generating...';
      try {
        const res = await api(`/api/local/companies/${currentCompany.id}/profile`, { method: 'POST' });
        const prof = res.profile || res.deterministic_fallback || {};
        document.getElementById('comp-profile-json').textContent = JSON.stringify(prof, null, 2);
      } catch (err) {
        alert(`Profile generation error: ${err.message}`);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Generate Business Profile';
        fetchSystemStatus();
      }
    });

    // Refresh Buttons
    document.getElementById('btn-refresh-company').addEventListener('click', loadCompanyAndProducts);
    document.getElementById('btn-refresh-telemetry').addEventListener('click', loadTelemetryAndQueue);
  }

  function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // =========================================================================
  // App Boot
  // =========================================================================
  window.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initControls();
    initPipelineButtons();
    initReplySandbox();

    // Initial data fetch
    fetchSystemStatus();
    loadCompanyAndProducts();
  });
})();
