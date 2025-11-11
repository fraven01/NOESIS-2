(function (globalObject, factory) {
  if (typeof module === 'object' && typeof module.exports === 'object') {
    module.exports = factory();
  } else {
    const target = globalObject || (typeof global !== 'undefined' ? global : {});
    target.DevHitlSupport = factory();
  }
})(typeof window !== 'undefined' ? window : this, function () {
  'use strict';

  const MAX_REASON_LENGTH = 280;

  function clampReason(reason) {
    if (typeof reason !== 'string') {
      return '';
    }
    const trimmed = reason.trim();
    if (trimmed.length <= MAX_REASON_LENGTH) {
      return trimmed;
    }
    return trimmed.slice(0, MAX_REASON_LENGTH);
  }

  function splitCustomUrls(value) {
    if (typeof value !== 'string') {
      return [];
    }
    return value
      .split(/\r?\n/)
      .map(function (entry) {
        return entry.trim();
      })
      .filter(Boolean);
  }

  function validateCustomUrls(urls) {
    if (!Array.isArray(urls)) {
      return { valid: [], invalid: [] };
    }
    const valid = [];
    const invalid = [];
    urls.forEach(function (entry) {
      if (isValidUrl(entry)) {
        valid.push(entry);
      } else {
        invalid.push(entry);
      }
    });
    return { valid: valid, invalid: invalid };
  }

  function isValidUrl(candidate) {
    if (typeof candidate !== 'string') {
      return false;
    }
    const trimmed = candidate.trim();
    if (!trimmed) {
      return false;
    }
    const lowered = trimmed.toLowerCase();
    if (!lowered.startsWith('http://') && !lowered.startsWith('https://')) {
      return false;
    }
    return true;
  }

  function formatCountdown(deadlineIso, nowDate) {
    if (!deadlineIso) {
      return { label: '–', overdue: false, remainingMs: 0 };
    }
    var parsedDeadline = new Date(deadlineIso);
    if (isNaN(parsedDeadline.valueOf())) {
      return { label: '–', overdue: false, remainingMs: 0 };
    }
    var reference = nowDate instanceof Date ? nowDate : new Date();
    var remaining = parsedDeadline.getTime() - reference.getTime();
    var overdue = remaining <= 0;
    var absolute = Math.abs(remaining);
    var minutes = Math.floor(absolute / 60000);
    var seconds = Math.floor((absolute % 60000) / 1000);
    var label = overdue ? 'Auto-approved' : 'Auto-Approve in ' + String(minutes).padStart(2, '0') + ':' + String(seconds).padStart(2, '0');
    return { label: label, overdue: overdue, remainingMs: remaining };
  }

  function computeDecisionSets(allIds, selectedIds, action) {
    var all = Array.isArray(allIds) ? allIds.filter(Boolean) : [];
    var selectedSet = new Set(Array.isArray(selectedIds) ? selectedIds : []);
    var approved = [];
    var rejected = [];

    if (action === 'approve_all') {
      approved = all.slice();
      rejected = [];
      return { approved: approved, rejected: rejected };
    }

    if (action === 'reject_all') {
      approved = [];
      rejected = all.slice();
      return { approved: approved, rejected: rejected };
    }

    if (action === 'approve_selected') {
      all.forEach(function (id) {
        if (selectedSet.has(id)) {
          approved.push(id);
        } else {
          rejected.push(id);
        }
      });
      return { approved: approved, rejected: rejected };
    }

    if (action === 'reject_selected') {
      all.forEach(function (id) {
        if (selectedSet.has(id)) {
          rejected.push(id);
        } else {
          approved.push(id);
        }
      });
      return { approved: approved, rejected: rejected };
    }

    return { approved: approved, rejected: rejected };
  }

  function parseJsonScript(elementId) {
    if (typeof document === 'undefined') {
      return null;
    }
    var element = document.getElementById(elementId);
    if (!element) {
      return null;
    }
    try {
      return JSON.parse(element.textContent || 'null');
    } catch (error) {
      return null;
    }
  }

  function normalizeFacetScores(facets) {
    if (!facets || typeof facets !== 'object') {
      return {};
    }
    var result = {};
    Object.keys(facets).forEach(function (key) {
      var value = facets[key];
      if (typeof value === 'number' && isFinite(value)) {
        result[key] = Math.round(value * 100) / 100;
      }
    });
    return result;
  }

  return {
    clampReason: clampReason,
    splitCustomUrls: splitCustomUrls,
    validateCustomUrls: validateCustomUrls,
    formatCountdown: formatCountdown,
    computeDecisionSets: computeDecisionSets,
    parseJsonScript: parseJsonScript,
    normalizeFacetScores: normalizeFacetScores,
  };
});
