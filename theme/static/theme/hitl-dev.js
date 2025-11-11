(function (globalObject) {
  if (!globalObject || !globalObject.document) {
    return;
  }

  var support = globalObject.DevHitlSupport;
  if (!support) {
    console.warn('DevHitlSupport helper is missing. HITL dev UI not initialised.');
    return;
  }

  var document = globalObject.document;
  var EVENT_SOURCE_RETRY_MS = 5000;
  var UNDO_TIMEOUT_MS = 5 * 60 * 1000;

  function DevHitlController(root) {
    this.root = root;
    this.dataset = root.dataset || {};
    this.runId = this.dataset.runId || '';
    this.devHeader = this.dataset.devHeader || '_DEV_ONLY_';
    this.devValue = this.dataset.devValue || 'true';
    this.state = {
      data: null,
      selected: new Set(),
      sortDirection: 'desc',
      submitting: false,
      undoPayload: null,
      undoDeadline: null,
      undoTimer: null,
      eventSource: null,
      eventSourceRetry: null,
      ingestionTasks: new Map(),
    };

    this.dom = {
      countdown: document.getElementById('hitl-countdown'),
      diversity: document.getElementById('hitl-diversity'),
      freshness: document.getElementById('hitl-freshness'),
      flags: document.getElementById('hitl-flags'),
      runLabel: document.getElementById('hitl-run-label'),
      tableContainer: document.getElementById('hitl-table-container'),
      customUrls: document.getElementById('hitl-custom-urls'),
      approveAll: document.getElementById('hitl-approve-all'),
      rejectAll: document.getElementById('hitl-reject-all'),
      approveSelected: document.getElementById('hitl-approve-selected'),
      rejectSelected: document.getElementById('hitl-reject-selected'),
      undoButton: document.getElementById('hitl-undo'),
      progressList: document.getElementById('hitl-progress-list'),
      progressSummary: document.getElementById('hitl-progress-summary'),
      coverageSummary: document.getElementById('hitl-coverage-summary'),
      coverageFacets: document.getElementById('hitl-coverage-facets'),
      coverageCounter: document.getElementById('hitl-coverage-counter'),
      successBanner: document.getElementById('hitl-success'),
      errorBanner: document.getElementById('hitl-error'),
    };

    this._countdownTimer = null;
    this._scoreHeaderElement = null;
    this._scoreHeaderListener = null;
    this._undoInterval = null;
    this._telemetry = [];
  }

  DevHitlController.prototype.init = function () {
    var initialData = support.parseJsonScript('hitl-initial-data');
    if (initialData) {
      this.state.data = initialData;
      this.render();
    }

    this._bindEvents();
    this._ensureEventSource();
    this._startCountdown();
  };

  DevHitlController.prototype._bindEvents = function () {
    var self = this;
    if (this.dom.approveAll) {
      this.dom.approveAll.addEventListener('click', function () {
        self._submitAction('approve_all');
      });
    }
    if (this.dom.rejectAll) {
      this.dom.rejectAll.addEventListener('click', function () {
        self._submitAction('reject_all');
      });
    }
    if (this.dom.approveSelected) {
      this.dom.approveSelected.addEventListener('click', function () {
        self._submitAction('approve_selected');
      });
    }
    if (this.dom.rejectSelected) {
      this.dom.rejectSelected.addEventListener('click', function () {
        self._submitAction('reject_selected');
      });
    }
    if (this.dom.undoButton) {
      this.dom.undoButton.addEventListener('click', function () {
        self._handleUndo();
      });
    }
    if (this.dom.customUrls) {
      this.dom.customUrls.addEventListener('input', function () {
        self._clearError();
        self._validateCustomUrls();
      });
    }

    document.addEventListener('keydown', function (event) {
      self._handleShortcut(event);
    });
  };

  DevHitlController.prototype._handleShortcut = function (event) {
    if (event.target && ['INPUT', 'TEXTAREA'].indexOf(event.target.tagName) >= 0) {
      return;
    }
    if (event.key && event.key.toLowerCase() === 'a' && !event.shiftKey && !event.metaKey && !event.ctrlKey) {
      event.preventDefault();
      this._selectAll(true);
    }
    if (event.key && event.key.toLowerCase() === 'a' && event.shiftKey) {
      event.preventDefault();
      this._selectAll(false);
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      this._submitAction('approve_selected');
    }
    if (event.key === 'Backspace') {
      event.preventDefault();
      this._submitAction('reject_selected');
    }
  };

  DevHitlController.prototype._selectAll = function (selectAll) {
    var ids = this._candidateIds();
    this.state.selected.clear();
    if (selectAll) {
      ids.forEach(this.state.selected.add, this.state.selected);
    }
    this._renderSelectionState();
  };

  DevHitlController.prototype.render = function () {
    if (!this.state.data) {
      return;
    }
    var data = this.state.data;
    this.runId = data.run_id || this.runId;
    if (this.dom.runLabel) {
      this.dom.runLabel.textContent = this.runId;
    }
    this._renderMeta(data.meta);
    this._renderCoverage(data.coverage_delta);
    this._renderTable(data.top_k || []);
    this._renderSelectionState();
  };

  DevHitlController.prototype._renderMeta = function (meta) {
    meta = meta || {};
    if (this.dom.diversity) {
      var text = meta.min_diversity_buckets ? 'Diversity ≥ ' + meta.min_diversity_buckets : '';
      this.dom.diversity.textContent = text;
    }
    if (this.dom.freshness) {
      var freshness = meta.freshness_mode ? 'Freshness: ' + meta.freshness_mode : '';
      this.dom.freshness.textContent = freshness;
    }
    if (this.dom.flags) {
      var badges = [];
      if (meta.rag_unavailable) {
        badges.push('RAG unavailable');
      }
      if (meta.llm_timeout) {
        badges.push('LLM fallback');
      }
      if (meta.cache_hit_rag) {
        badges.push('RAG cache hit');
      }
      if (meta.cache_hit_llm) {
        badges.push('LLM cache hit');
      }
      if (meta.auto_approved) {
        badges.push('Auto-approved');
      }
      this.dom.flags.textContent = badges.join(' · ');
    }
  };

  DevHitlController.prototype._renderCoverage = function (coverage) {
    coverage = coverage || {};
    if (this.dom.coverageSummary) {
      this.dom.coverageSummary.textContent = coverage.summary || 'Keine Coverage-Details vorhanden.';
    }
    var before = support.normalizeFacetScores(coverage.facets_before || {});
    var after = support.normalizeFacetScores(coverage.facets_after || {});
    if (this.dom.coverageFacets) {
      this.dom.coverageFacets.innerHTML = '';
      var keys = Object.keys(after);
      if (keys.length === 0) {
        var placeholder = document.createElement('p');
        placeholder.className = 'text-xs text-slate-500';
        placeholder.textContent = 'Keine Facetten verfügbar.';
        this.dom.coverageFacets.appendChild(placeholder);
      } else {
        keys.forEach(function (key) {
          var wrapper = document.createElement('div');
          wrapper.className = 'flex items-center justify-between rounded bg-slate-50 px-3 py-2';
          var label = document.createElement('dt');
          label.className = 'font-medium text-slate-700';
          label.textContent = key;
          var value = document.createElement('dd');
          value.className = 'text-slate-600';
          var beforeValue = before[key] !== undefined ? before[key].toFixed(2) : '–';
          var afterValue = after[key] !== undefined ? after[key].toFixed(2) : '–';
          value.textContent = beforeValue + ' → ' + afterValue;
          wrapper.appendChild(label);
          wrapper.appendChild(value);
          this.dom.coverageFacets.appendChild(wrapper);
        }, this);
      }
    }

    if (this.dom.coverageCounter) {
      var meta = this.state.data ? this.state.data.meta || {} : {};
      var ingested = meta.ingested_count || 0;
      var total = meta.total_candidates || (this.state.data && this.state.data.top_k ? this.state.data.top_k.length : 0);
      this.dom.coverageCounter.textContent = 'Ingested ' + ingested + '/' + total;
    }
  };

  DevHitlController.prototype._renderTable = function (items) {
    var container = this.dom.tableContainer;
    if (!container) {
      return;
    }
    container.innerHTML = '';
    if (!items || !items.length) {
      var empty = document.createElement('p');
      empty.className = 'text-sm text-slate-500';
      empty.textContent = 'Keine Kandidaten für diesen Run.';
      container.appendChild(empty);
      return;
    }

    var data = items.slice();
    if (this.state.sortDirection === 'asc') {
      data.sort(function (a, b) {
        return (a.fused_score || 0) - (b.fused_score || 0);
      });
    } else {
      data.sort(function (a, b) {
        return (b.fused_score || 0) - (a.fused_score || 0);
      });
    }

    var wrapper = document.createElement('div');
    wrapper.className = 'overflow-x-auto rounded-lg border border-slate-200';
    var table = document.createElement('table');
    table.className = 'min-w-full divide-y divide-slate-200 text-left text-sm';

    var thead = document.createElement('thead');
    thead.className = 'bg-slate-50 text-xs uppercase tracking-wide text-slate-600';
    var headerRow = document.createElement('tr');

    headerRow.appendChild(this._buildHeaderCell('Kandidaten', { scope: 'col', sr: true }));

    headerRow.appendChild(this._buildHeaderCell('Titel & Quelle', { scope: 'col' }));

    var scoreHeader = this._buildHeaderCell('Fused Score', { scope: 'col', sortable: true });
    headerRow.appendChild(scoreHeader);

    headerRow.appendChild(this._buildHeaderCell('Reason', { scope: 'col' }));
    headerRow.appendChild(this._buildHeaderCell('Gap Tags', { scope: 'col' }));
    headerRow.appendChild(this._buildHeaderCell('Risk Flags', { scope: 'col' }));

    thead.appendChild(headerRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    tbody.className = 'divide-y divide-slate-200 bg-white';

    var self = this;
    data.forEach(function (item, index) {
      var row = document.createElement('tr');
      row.className = index % 2 === 0 ? 'bg-white' : 'bg-slate-50';
      row.dataset.candidateId = item.id;

      var selectCell = document.createElement('td');
      selectCell.className = 'w-12 px-3 py-3 align-top';
      var checkboxId = 'hitl-select-' + index;
      var checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.id = checkboxId;
      checkbox.value = item.id;
      checkbox.className = 'h-4 w-4 rounded border-slate-300 text-slate-900 focus:ring-slate-500';
      checkbox.checked = self.state.selected.has(item.id);
      checkbox.setAttribute('aria-label', 'Kandidat ' + (item.title || item.url || item.id));
      checkbox.addEventListener('change', function () {
        if (checkbox.checked) {
          self.state.selected.add(item.id);
        } else {
          self.state.selected.delete(item.id);
        }
        self._renderSelectionState();
      });
      selectCell.appendChild(checkbox);
      row.appendChild(selectCell);

      var titleCell = document.createElement('td');
      titleCell.className = 'max-w-xs px-3 py-3 align-top';
      var titleLink = document.createElement('a');
      titleLink.href = item.url || '#';
      titleLink.target = '_blank';
      titleLink.rel = 'noopener noreferrer';
      titleLink.className = 'text-sm font-semibold text-slate-900 underline-offset-2 hover:underline';
      titleLink.textContent = item.title || item.url || item.id;
      titleCell.appendChild(titleLink);

      var metaList = document.createElement('div');
      metaList.className = 'mt-1 space-y-1 text-xs text-slate-600';
      if (item.domain) {
        var domain = document.createElement('p');
        domain.textContent = item.domain;
        metaList.appendChild(domain);
      }
      if (item.source) {
        var source = document.createElement('p');
        source.textContent = 'Source: ' + item.source;
        metaList.appendChild(source);
      }
      if (item.detected_date) {
        var detected = document.createElement('p');
        detected.textContent = 'Detected: ' + item.detected_date;
        metaList.appendChild(detected);
      }
      if (item.version_hint) {
        var version = document.createElement('p');
        version.textContent = 'Version: ' + item.version_hint;
        metaList.appendChild(version);
      }
      titleCell.appendChild(metaList);
      row.appendChild(titleCell);

      var scoreCell = document.createElement('td');
      scoreCell.className = 'w-32 px-3 py-3 align-top';
      var fusedScore = document.createElement('p');
      fusedScore.className = 'text-sm font-semibold text-slate-900';
      fusedScore.textContent = typeof item.fused_score === 'number' ? item.fused_score.toFixed(3) : '–';
      scoreCell.appendChild(fusedScore);
      var llmScore = document.createElement('p');
      llmScore.className = 'text-xs text-slate-600';
      llmScore.textContent = 'LLM Score: ' + (item.score !== undefined ? item.score : '–');
      scoreCell.appendChild(llmScore);
      row.appendChild(scoreCell);

      var reasonCell = document.createElement('td');
      reasonCell.className = 'max-w-lg px-3 py-3 align-top';
      var reasonWrapper = document.createElement('div');
      reasonWrapper.className = 'space-y-2';
      var reasonText = document.createElement('p');
      reasonText.className = 'text-sm text-slate-700';
      var clamped = support.clampReason(item.reason || '');
      var truncated = clamped.length < (item.reason || '').length;
      reasonText.textContent = clamped;
      reasonWrapper.appendChild(reasonText);
      if (truncated) {
        var disclosure = document.createElement('button');
        disclosure.type = 'button';
        disclosure.className = 'text-xs font-medium text-slate-600 underline-offset-2 hover:underline';
        disclosure.textContent = 'mehr…';
        disclosure.addEventListener('click', function () {
          reasonText.textContent = item.reason || '';
          disclosure.remove();
        });
        reasonWrapper.appendChild(disclosure);
      }
      reasonCell.appendChild(reasonWrapper);
      row.appendChild(reasonCell);

      var gapCell = document.createElement('td');
      gapCell.className = 'w-48 px-3 py-3 align-top';
      gapCell.appendChild(self._renderBadges(item.gap_tags || [], 'bg-emerald-50 text-emerald-800 border border-emerald-200'));
      row.appendChild(gapCell);

      var riskCell = document.createElement('td');
      riskCell.className = 'w-48 px-3 py-3 align-top';
      riskCell.appendChild(self._renderBadges(item.risk_flags || [], 'bg-rose-50 text-rose-800 border border-rose-200'));
      row.appendChild(riskCell);

      tbody.appendChild(row);
    });

    table.appendChild(tbody);
    wrapper.appendChild(table);
    container.appendChild(wrapper);

    var self = this;
    if (this._scoreHeaderElement && this._scoreHeaderListener) {
      this._scoreHeaderElement.removeEventListener('click', this._scoreHeaderListener);
    }
    this._scoreHeaderListener = function () {
      self.state.sortDirection = self.state.sortDirection === 'asc' ? 'desc' : 'asc';
      self._renderTable(self.state.data.top_k || []);
    };
    this._scoreHeaderElement = scoreHeader;
    scoreHeader.addEventListener('click', this._scoreHeaderListener);
  };

  DevHitlController.prototype._renderBadges = function (values, className) {
    var container = document.createElement('div');
    container.className = 'flex flex-wrap gap-1';
    if (!values.length) {
      var placeholder = document.createElement('span');
      placeholder.className = 'text-xs text-slate-500';
      placeholder.textContent = '–';
      container.appendChild(placeholder);
      return container;
    }
    values.forEach(function (value) {
      var badge = document.createElement('span');
      badge.className = 'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ' + className;
      badge.textContent = value;
      badge.setAttribute('role', 'status');
      container.appendChild(badge);
    });
    return container;
  };

  DevHitlController.prototype._buildHeaderCell = function (label, options) {
    options = options || {};
    var cell = document.createElement('th');
    cell.className = 'px-3 py-3 text-left font-semibold';
    if (options.scope) {
      cell.scope = options.scope;
    }
    if (options.sr) {
      var span = document.createElement('span');
      span.className = 'sr-only';
      span.textContent = label;
      cell.appendChild(span);
      return cell;
    }
    if (options.sortable) {
      cell.className += ' cursor-pointer select-none';
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'inline-flex items-center gap-1 text-xs font-semibold uppercase tracking-wide text-slate-600';
      var arrow = this.state.sortDirection === 'asc' ? '↑' : '↓';
      button.textContent = label + ' ' + arrow;
      cell.appendChild(button);
    } else {
      cell.textContent = label;
    }
    return cell;
  };

  DevHitlController.prototype._renderSelectionState = function () {
    var count = this.state.selected.size;
    if (this.dom.flags) {
      var baseText = this.dom.flags.textContent || '';
      var segments = baseText.split(' · ').filter(Boolean);
      segments = segments.filter(function (segment) {
        return !segment.startsWith('Selected ');
      });
      if (count > 0) {
        segments.unshift('Selected ' + count);
      }
      this.dom.flags.textContent = segments.join(' · ');
    }
    var checkboxes = this.dom.tableContainer ? this.dom.tableContainer.querySelectorAll('input[type="checkbox"]') : [];
    Array.prototype.forEach.call(checkboxes, function (checkbox) {
      var id = checkbox.value;
      checkbox.checked = this.state.selected.has(id);
    }, this);
  };

  DevHitlController.prototype._submitAction = function (action) {
    if (this.state.submitting) {
      return;
    }
    var ids = this._candidateIds();
    if (!ids.length) {
      return;
    }

    if ((action === 'approve_selected' || action === 'reject_selected') && this.state.selected.size === 0) {
      this._showError('Bitte wähle mindestens einen Kandidaten aus.');
      return;
    }

    var decision = support.computeDecisionSets(ids, Array.from(this.state.selected), action);
    var urls = this._validateCustomUrls();
    if (!urls) {
      return;
    }
    var payload = {
      run_id: this.runId,
      approved_ids: decision.approved,
      rejected_ids: decision.rejected,
      custom_urls: urls.valid,
    };
    this._sendSubmission(action, payload, urls.invalid);
  };

  DevHitlController.prototype._validateCustomUrls = function () {
    if (!this.dom.customUrls) {
      return { valid: [], invalid: [] };
    }
    var urls = support.splitCustomUrls(this.dom.customUrls.value || '');
    var result = support.validateCustomUrls(urls);
    if (result.invalid.length) {
      this.dom.customUrls.classList.add('border-red-400', 'focus:ring-red-500');
      this._showError('Ungültige URLs: ' + result.invalid.join(', '));
      return null;
    }
    this.dom.customUrls.classList.remove('border-red-400', 'focus:ring-red-500');
    return result;
  };

  DevHitlController.prototype._sendSubmission = function (action, payload, invalidUrls) {
    var self = this;
    this._clearError();
    this._showSuccess('Sende Entscheidung …');
    this.state.submitting = true;
    this._logTelemetry('ui.action', {
      action: action,
      selection_count: this.state.selected.size,
      custom_url_count: payload.custom_urls.length,
      invalid_urls: invalidUrls || [],
    });

    var headers = {
      'Content-Type': 'application/json',
    };
    headers[this.devHeader] = this.devValue;
    var csrfToken = this._readCsrfToken();
    if (csrfToken) {
      headers['X-CSRFToken'] = csrfToken;
    }

    globalObject.fetch('/dev/hitl/approve-candidates/', {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(payload),
      credentials: 'same-origin',
    })
      .then(function (response) {
        if (!response.ok) {
          return response.json().catch(function () {
            return { error: 'http_error' };
          }).then(function (body) {
            throw body;
          });
        }
        return response.json();
      })
      .then(function (body) {
        self._handleSubmissionSuccess(payload, body);
      })
      .catch(function (error) {
        var message = error && error.error ? error.error : 'Unbekannter Fehler beim Senden.';
        self._showError(message);
      })
      .finally(function () {
        self.state.submitting = false;
      });
  };

  DevHitlController.prototype._handleSubmissionSuccess = function (payload, response) {
    this._showSuccess('Freigabe übermittelt.');
    this.state.selected.clear();
    this._renderSelectionState();
    this._ensureEventSource(true);
    this._registerUndo(payload);
    if (this.dom.customUrls) {
      this.dom.customUrls.value = '';
    }
    if (response && Array.isArray(response.ingestion_task_ids)) {
      var self = this;
      response.ingestion_task_ids.forEach(function (taskId) {
        self.state.ingestionTasks.set(taskId, { status: 'queued', url: null });
      });
      this._renderIngestionTasks();
    }
  };

  DevHitlController.prototype._registerUndo = function (payload) {
    var inverse = {
      run_id: payload.run_id,
      approved_ids: payload.rejected_ids.slice(),
      rejected_ids: payload.approved_ids.slice(),
      custom_urls: [],
    };
    this._clearUndoTimers();
    this.state.undoPayload = inverse;
    this.state.undoDeadline = Date.now() + UNDO_TIMEOUT_MS;
    this._updateUndoState();
    var self = this;
    this.state.undoTimer = setTimeout(function () {
      self.state.undoPayload = null;
      self.state.undoDeadline = null;
      self._clearUndoTimers();
      self._updateUndoState();
    }, UNDO_TIMEOUT_MS);
    this._undoInterval = setInterval(function () {
      self._updateUndoState();
    }, 1000);
  };

  DevHitlController.prototype._handleUndo = function () {
    if (!this.state.undoPayload) {
      return;
    }
    this._sendSubmission('undo', this.state.undoPayload, []);
    this.state.undoPayload = null;
    this.state.undoDeadline = null;
    this._clearUndoTimers();
    this._updateUndoState();
  };

  DevHitlController.prototype._updateUndoState = function () {
    if (!this.dom.undoButton) {
      return;
    }
    if (this.state.undoPayload) {
      this.dom.undoButton.disabled = false;
      this.dom.undoButton.setAttribute('aria-live', 'polite');
      this.dom.undoButton.textContent = '🔄 Undo (' + this._formatUndoCountdown() + ')';
    } else {
      this.dom.undoButton.disabled = true;
      this.dom.undoButton.removeAttribute('aria-live');
      this.dom.undoButton.textContent = '🔄 Undo';
    }
  };

  DevHitlController.prototype._formatUndoCountdown = function () {
    if (!this.state.undoDeadline) {
      return '0:00';
    }
    var remaining = Math.max(this.state.undoDeadline - Date.now(), 0);
    var minutes = Math.floor(remaining / 60000);
    var seconds = Math.floor((remaining % 60000) / 1000);
    return String(minutes) + ':' + String(seconds).padStart(2, '0');
  };

  DevHitlController.prototype._ensureEventSource = function (forceRestart) {
    var self = this;
    if (forceRestart && this.state.eventSource) {
      this.state.eventSource.close();
      this.state.eventSource = null;
    }
    if (this.state.eventSource) {
      return;
    }
    if (!globalObject.EventSource) {
      console.warn('EventSource not supported in this browser.');
      return;
    }
    var url = '/dev/hitl/progress/' + encodeURIComponent(this.runId) + '/stream/?dev_token=true';
    var source = new globalObject.EventSource(url, { withCredentials: true });
    source.addEventListener('ingestion_update', function (event) {
      self._handleIngestionUpdate(event);
    });
    source.addEventListener('coverage_update', function (event) {
      self._handleCoverageUpdate(event);
    });
    source.addEventListener('deadline_update', function (event) {
      self._handleDeadlineUpdate(event);
    });
    source.onerror = function () {
      source.close();
      self.state.eventSource = null;
      if (self.state.eventSourceRetry) {
        clearTimeout(self.state.eventSourceRetry);
      }
      self.state.eventSourceRetry = setTimeout(function () {
        self._ensureEventSource(true);
      }, EVENT_SOURCE_RETRY_MS);
    };
    this.state.eventSource = source;
  };

  DevHitlController.prototype._handleIngestionUpdate = function (event) {
    try {
      var payload = JSON.parse(event.data || '{}');
      if (!payload.task_id) {
        return;
      }
      var task = this.state.ingestionTasks.get(payload.task_id) || {};
      task.status = payload.status || task.status || 'queued';
      if (payload.url) {
        task.url = payload.url;
      }
      this.state.ingestionTasks.set(payload.task_id, task);
      this._renderIngestionTasks();
    } catch (error) {
      console.error('Failed to parse ingestion update', error);
    }
  };

  DevHitlController.prototype._renderIngestionTasks = function () {
    if (!this.dom.progressList) {
      return;
    }
    var entries = Array.from(this.state.ingestionTasks.entries());
    this.dom.progressList.innerHTML = '';
    if (!entries.length) {
      var placeholder = document.createElement('li');
      placeholder.className = 'px-4 py-3 text-sm text-slate-500';
      placeholder.textContent = 'Keine Aufgaben gestartet.';
      this.dom.progressList.appendChild(placeholder);
      return;
    }
    var counters = { queued: 0, running: 0, done: 0, failed: 0 };
    entries.forEach(function (entry) {
      var taskId = entry[0];
      var info = entry[1] || {};
      var item = document.createElement('li');
      item.className = 'px-4 py-3 text-sm text-slate-700';
      var header = document.createElement('div');
      header.className = 'flex items-center justify-between gap-4';
      var label = document.createElement('span');
      label.className = 'font-medium text-slate-900';
      label.textContent = info.url || taskId;
      header.appendChild(label);

      var status = document.createElement('span');
      status.className = 'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold';
      var statusText = info.status || 'queued';
      counters[statusText] = (counters[statusText] || 0) + 1;
      if (statusText === 'done') {
        status.className += ' bg-emerald-50 text-emerald-800';
      } else if (statusText === 'running') {
        status.className += ' bg-amber-50 text-amber-800';
      } else if (statusText === 'failed') {
        status.className += ' bg-rose-50 text-rose-800';
      } else {
        status.className += ' bg-slate-100 text-slate-600';
      }
      status.textContent = statusText;
      header.appendChild(status);
      item.appendChild(header);

      var meta = document.createElement('p');
      meta.className = 'mt-1 text-xs text-slate-500';
      meta.textContent = 'Task ' + taskId;
      item.appendChild(meta);

      this.dom.progressList.appendChild(item);
    }, this);

    if (this.dom.progressSummary) {
      var summary = [];
      if (counters.running) {
        summary.push(counters.running + ' running');
      }
      if (counters.queued) {
        summary.push(counters.queued + ' queued');
      }
      if (counters.done) {
        summary.push(counters.done + ' done');
      }
      if (counters.failed) {
        summary.push(counters.failed + ' failed');
      }
      this.dom.progressSummary.textContent = summary.join(' · ');
    }
  };

  DevHitlController.prototype._handleCoverageUpdate = function (event) {
    try {
      var payload = JSON.parse(event.data || '{}');
      if (this.state.data && this.state.data.meta) {
        this.state.data.meta.ingested_count = payload.ingested_count || this.state.data.meta.ingested_count;
        this.state.data.meta.total_candidates = payload.total || this.state.data.meta.total_candidates;
      }
      if (this.state.data && this.state.data.coverage_delta) {
        this.state.data.coverage_delta.facets_after = payload.facets_after || this.state.data.coverage_delta.facets_after;
      }
      this._renderCoverage(this.state.data.coverage_delta);
    } catch (error) {
      console.error('Failed to parse coverage update', error);
    }
  };

  DevHitlController.prototype._handleDeadlineUpdate = function (event) {
    try {
      var payload = JSON.parse(event.data || '{}');
      if (!this.state.data) {
        return;
      }
      this.state.data.meta = this.state.data.meta || {};
      if (payload.deadline_utc) {
        this.state.data.meta.deadline_utc = payload.deadline_utc;
      }
      if (typeof payload.auto_approved === 'boolean') {
        this.state.data.meta.auto_approved = payload.auto_approved;
      }
      this._startCountdown();
    } catch (error) {
      console.error('Failed to parse deadline update', error);
    }
  };

  DevHitlController.prototype._startCountdown = function () {
    var self = this;
    if (this._countdownTimer) {
      clearInterval(this._countdownTimer);
    }
    var update = function () {
      if (!self.dom.countdown || !self.state.data) {
        return;
      }
      var deadline = self.state.data.meta ? self.state.data.meta.deadline_utc : null;
      var result = support.formatCountdown(deadline);
      self.dom.countdown.textContent = result.label;
      if (result.overdue) {
        self.dom.countdown.classList.remove('bg-blue-50', 'text-blue-700');
        self.dom.countdown.classList.add('bg-emerald-50', 'text-emerald-800');
      } else {
        self.dom.countdown.classList.add('bg-blue-50', 'text-blue-700');
        self.dom.countdown.classList.remove('bg-emerald-50', 'text-emerald-800');
      }
    };
    update();
    this._countdownTimer = setInterval(update, 1000);
  };

  DevHitlController.prototype._clearUndoTimers = function () {
    if (this.state.undoTimer) {
      clearTimeout(this.state.undoTimer);
      this.state.undoTimer = null;
    }
    if (this._undoInterval) {
      clearInterval(this._undoInterval);
      this._undoInterval = null;
    }
  };

  DevHitlController.prototype._candidateIds = function () {
    return this.state.data && Array.isArray(this.state.data.top_k)
      ? this.state.data.top_k.map(function (item) {
          return item.id;
        })
      : [];
  };

  DevHitlController.prototype._showError = function (message) {
    if (this.dom.errorBanner) {
      this.dom.errorBanner.textContent = message;
      this.dom.errorBanner.classList.remove('hidden');
    }
    if (this.dom.successBanner) {
      this.dom.successBanner.classList.add('hidden');
    }
  };

  DevHitlController.prototype._showSuccess = function (message) {
    if (this.dom.successBanner) {
      this.dom.successBanner.textContent = message;
      this.dom.successBanner.classList.remove('hidden');
    }
    if (this.dom.errorBanner) {
      this.dom.errorBanner.classList.add('hidden');
    }
  };

  DevHitlController.prototype._clearError = function () {
    if (this.dom.errorBanner) {
      this.dom.errorBanner.classList.add('hidden');
    }
  };

  DevHitlController.prototype._readCsrfToken = function () {
    var name = 'csrftoken=';
    var cookies = document.cookie ? document.cookie.split(';') : [];
    for (var i = 0; i < cookies.length; i += 1) {
      var cookie = cookies[i].trim();
      if (cookie.substring(0, name.length) === name) {
        return decodeURIComponent(cookie.substring(name.length));
      }
    }
    return null;
  };

  DevHitlController.prototype._logTelemetry = function (name, payload) {
    try {
      this._telemetry.push({ name: name, payload: payload, ts: Date.now() });
      if (console && console.info) {
        console.info('hitl.dev.telemetry', name, payload);
      }
    } catch (error) {
      // ignore telemetry errors
    }
  };

  globalObject.addEventListener('beforeunload', function () {
    if (controller && controller.state && controller.state.eventSource) {
      controller.state.eventSource.close();
    }
  });

  var controller = null;

  document.addEventListener('DOMContentLoaded', function () {
    var root = document.querySelector('[data-hitl-root]');
    if (!root) {
      return;
    }
    controller = new DevHitlController(root);
    controller.init();
  });
})(typeof window !== 'undefined' ? window : undefined);
