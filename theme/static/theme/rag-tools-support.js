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

  const crawlerModeMap = {
    manual: 'manual',
    live: 'live',
    ingest: 'live',
    store_only: 'store_only',
    fetch_only: 'fetch_only',
  };

  function mapCrawlerMode(value) {
    if (typeof value !== 'string') {
      return null;
    }
    const lowered = value.trim().toLowerCase();
    if (!lowered) {
      return null;
    }
    return crawlerModeMap[lowered] || 'live';
  }

  function buildCrawlerPayload(options) {
    const opts = options || {};
    const payload = {};

    const parseInteger = function (value) {
      if (value === undefined || value === null || value === '') {
        return null;
      }
      const numeric = Number(value);
      if (!Number.isFinite(numeric) || numeric < 0) {
        return null;
      }
      return Math.floor(numeric);
    };

    if (opts.workflowId) {
      payload.workflow_id = String(opts.workflowId).trim();
    }
    if (opts.mode) {
      const mappedMode = mapCrawlerMode(opts.mode);
      if (mappedMode) {
        payload.mode = mappedMode;
      }
    }
    if (opts.originUrl) {
      const origin = String(opts.originUrl).trim();
      if (origin) {
        payload.origin_url = origin;
      }
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

    const hasFetchFlag = Object.prototype.hasOwnProperty.call(opts, 'fetch');
    if (hasFetchFlag) {
      payload.fetch = Boolean(opts.fetch);
    }

    const contentProvided =
      Object.prototype.hasOwnProperty.call(opts, 'content') &&
      typeof opts.content === 'string' &&
      opts.content.length > 0;
    if (contentProvided) {
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

    const limitsPayload = {};
    const parsedMaxDocumentBytes = parseInteger(opts.maxDocumentBytes);
    if (parsedMaxDocumentBytes !== null) {
      payload.max_document_bytes = parsedMaxDocumentBytes;
      limitsPayload.max_document_bytes = parsedMaxDocumentBytes;
    }
    const providedLimits = opts.limits && typeof opts.limits === 'object' ? opts.limits : null;
    if (providedLimits) {
      ['maxDocumentBytes', 'max_document_bytes'].forEach(function (key) {
        if (!Object.prototype.hasOwnProperty.call(providedLimits, key)) {
          return;
        }
        const parsed = parseInteger(providedLimits[key]);
        if (parsed !== null) {
          limitsPayload.max_document_bytes = parsed;
        }
      });
    }
    if (Object.keys(limitsPayload).length) {
      payload.limits = limitsPayload;
    }

    if (Object.prototype.hasOwnProperty.call(opts, 'shadowMode')) {
      payload.shadow_mode = Boolean(opts.shadowMode);
    }
    if (Object.prototype.hasOwnProperty.call(opts, 'dryRun')) {
      payload.dry_run = Boolean(opts.dryRun);
    }

    let snapshotOptions = null;
    if (Object.prototype.hasOwnProperty.call(opts, 'snapshot')) {
      snapshotOptions = { enabled: Boolean(opts.snapshot) };
    }
    const snapshotLabel = Object.prototype.hasOwnProperty.call(opts, 'snapshotLabel')
      ? opts.snapshotLabel
      : opts.snapshot_label;
    if (typeof snapshotLabel === 'string' && snapshotLabel.trim()) {
      const trimmedSnapshotLabel = snapshotLabel.trim();
      payload.snapshot_label = trimmedSnapshotLabel;
      if (!snapshotOptions) {
        snapshotOptions = {};
      }
      snapshotOptions.label = trimmedSnapshotLabel;
    }
    if (snapshotOptions) {
      payload.snapshot = snapshotOptions;
    }

    const reviewRaw = Object.prototype.hasOwnProperty.call(opts, 'review') ? opts.review : undefined;
    const normalizedReview = normalizeCrawlerManualReview(
      typeof reviewRaw === 'string' ? reviewRaw : ''
    );
    const manualRaw = Object.prototype.hasOwnProperty.call(opts, 'manualReview')
      ? opts.manualReview
      : opts.manual_review;
    const normalizedManual = normalizeCrawlerManualReview(
      typeof manualRaw === 'string' ? manualRaw : ''
    );
    if (normalizedReview) {
      payload.review = normalizedReview;
    }
    if (normalizedManual) {
      payload.manual_review = normalizedManual;
      if (!payload.review) {
        payload.review = normalizedManual;
      }
    } else if (payload.review) {
      payload.manual_review = payload.review;
    }

    if (opts.collectionId) {
      const collectionId = String(opts.collectionId).trim();
      if (collectionId) {
        payload.collection_id = collectionId;
      }
    }

    if (Object.prototype.hasOwnProperty.call(opts, 'forceRetire')) {
      payload.force_retire = Boolean(opts.forceRetire);
    }
    if (Object.prototype.hasOwnProperty.call(opts, 'recomputeDelta')) {
      payload.recompute_delta = Boolean(opts.recomputeDelta);
    }

    let originUrls = [];
    if (Array.isArray(opts.originUrls)) {
      originUrls = opts.originUrls
        .map(function (entry) {
          if (typeof entry === 'string') {
            return entry.trim();
          }
          return String(entry || '').trim();
        })
        .filter(Boolean);
    } else if (typeof opts.originUrls === 'string') {
      originUrls = opts.originUrls
        .split(/[\n,]/)
        .map(function (entry) {
          return entry.trim();
        })
        .filter(Boolean);
    }
    const seenOrigins = new Set();
    const combinedOrigins = [];
    if (payload.origin_url) {
      const primaryOrigin = payload.origin_url;
      if (!seenOrigins.has(primaryOrigin)) {
        seenOrigins.add(primaryOrigin);
        combinedOrigins.push(primaryOrigin);
      }
    }
    originUrls.forEach(function (url) {
      if (!seenOrigins.has(url)) {
        seenOrigins.add(url);
        combinedOrigins.push(url);
      }
    });
    if (combinedOrigins.length) {
      payload.origins = combinedOrigins.map(function (url) {
        const originEntry = { url: url };
        if (payload.review) {
          originEntry.review = payload.review;
        }
        if (Object.prototype.hasOwnProperty.call(opts, 'dryRun')) {
          originEntry.dry_run = Boolean(opts.dryRun);
        }
        if (payload.limits && Object.prototype.hasOwnProperty.call(payload.limits, 'max_document_bytes')) {
          originEntry.limits = { max_document_bytes: payload.limits.max_document_bytes };
        }
        if (snapshotOptions) {
          const perOriginSnapshot = {};
          if (Object.prototype.hasOwnProperty.call(snapshotOptions, 'enabled')) {
            perOriginSnapshot.enabled = Boolean(snapshotOptions.enabled);
          }
          if (snapshotOptions.label) {
            perOriginSnapshot.label = snapshotOptions.label;
          }
          if (Object.keys(perOriginSnapshot).length) {
            originEntry.snapshot = perOriginSnapshot;
          }
        }
        return originEntry;
      });
    }

    if (hasFetchFlag && payload.fetch === false && !Object.prototype.hasOwnProperty.call(payload, 'content')) {
      payload.content = '';
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
