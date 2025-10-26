(function (globalObject, factory) {
  if (typeof module === 'object' && typeof module.exports === 'object') {
    module.exports = factory();
  } else {
    const target = globalObject || (typeof global !== 'undefined' ? global : {});
    target.RagToolsSupport = factory();
  }
})(typeof window !== 'undefined' ? window : this, function () {
  'use strict';

  function normalizeCollectionInput(value) {
    if (typeof value !== 'string') {
      return '';
    }
    const trimmed = value.trim();
    return trimmed || '';
  }

  function resolveEffectiveCollectionId(options) {
    const opts = options || {};
    const normalizedInput = normalizeCollectionInput(opts.collectionInput);
    if (normalizedInput) {
      return normalizedInput;
    }
    if (!opts.allowLegacy) {
      return '';
    }
    return normalizeCollectionInput(opts.legacyDocClass);
  }

  function sanitizeKnownCollections(values) {
    if (!Array.isArray(values)) {
      return [];
    }
    const seen = new Set();
    const result = [];
    values.forEach(function (value) {
      const normalized = normalizeCollectionInput(value);
      if (!normalized || seen.has(normalized)) {
        return;
      }
      seen.add(normalized);
      result.push(normalized);
    });
    return result;
  }

  function isKnownCollection(value, knownCollections) {
    if (!Array.isArray(knownCollections) || knownCollections.length === 0) {
      return false;
    }
    const normalizedValue = normalizeCollectionInput(value);
    if (!normalizedValue) {
      return false;
    }
    return knownCollections.includes(normalizedValue);
  }

  function formatCollectionDisplay(value) {
    const normalized = normalizeCollectionInput(value);
    return normalized || '–';
  }

  function normalizeCrawlerManualReview(value) {
    if (typeof value !== 'string') {
      return null;
    }
    const lowered = value.trim().toLowerCase();
    if (!lowered) {
      return null;
    }
    if (lowered === 'required' || lowered === 'approved' || lowered === 'rejected') {
      return lowered;
    }
    return null;
  }

  function buildCrawlerPayload(options) {
    const opts = options || {};
    const payload = {};
    if (opts.workflowId) {
      payload.workflow_id = String(opts.workflowId).trim();
    }
    if (opts.originUrl) {
      payload.origin_url = String(opts.originUrl).trim();
    }
    if (opts.documentId) {
      payload.document_id = String(opts.documentId).trim();
    }
    if (opts.title) {
      payload.title = String(opts.title).trim();
    }
    if (opts.language) {
      payload.language = String(opts.language).trim();
    }
    if (opts.provider) {
      payload.provider = String(opts.provider).trim();
    }
    if (opts.contentType) {
      payload.content_type = String(opts.contentType).trim();
    }
    if (opts.content) {
      payload.content = String(opts.content);
    }
    if (opts.tags) {
      if (Array.isArray(opts.tags)) {
        payload.tags = opts.tags
          .map(function (tag) {
            return typeof tag === 'string' ? tag.trim() : '';
          })
          .filter(Boolean);
      } else if (typeof opts.tags === 'string') {
        payload.tags = opts.tags
          .split(',')
          .map(function (entry) {
            return entry.trim();
          })
          .filter(Boolean);
      }
    }
    if (
      opts.maxDocumentBytes !== undefined &&
      opts.maxDocumentBytes !== null &&
      opts.maxDocumentBytes !== ''
    ) {
      const parsed = Number(opts.maxDocumentBytes);
      if (!Number.isNaN(parsed) && parsed >= 0) {
        payload.max_document_bytes = Math.floor(parsed);
      }
    }
    if (Object.prototype.hasOwnProperty.call(opts, 'shadowMode')) {
      payload.shadow_mode = Boolean(opts.shadowMode);
    }
    if (Object.prototype.hasOwnProperty.call(opts, 'dryRun')) {
      payload.dry_run = Boolean(opts.dryRun);
    }
    const manual = Object.prototype.hasOwnProperty.call(opts, 'manualReview')
      ? opts.manualReview
      : opts.manual_review;
    const normalizedManual = normalizeCrawlerManualReview(
      typeof manual === 'string' ? manual : ''
    );
    if (normalizedManual) {
      payload.manual_review = normalizedManual;
    }
    if (Object.prototype.hasOwnProperty.call(opts, 'forceRetire')) {
      payload.force_retire = Boolean(opts.forceRetire);
    }
    if (Object.prototype.hasOwnProperty.call(opts, 'recomputeDelta')) {
      payload.recompute_delta = Boolean(opts.recomputeDelta);
    }
    return payload;
  }

  return {
    normalizeCollectionInput: normalizeCollectionInput,
    resolveEffectiveCollectionId: resolveEffectiveCollectionId,
    sanitizeKnownCollections: sanitizeKnownCollections,
    isKnownCollection: isKnownCollection,
    formatCollectionDisplay: formatCollectionDisplay,
    buildCrawlerPayload: buildCrawlerPayload,
    normalizeCrawlerManualReview: normalizeCrawlerManualReview,
  };
});
