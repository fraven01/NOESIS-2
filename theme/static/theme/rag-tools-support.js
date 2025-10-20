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

  return {
    normalizeCollectionInput: normalizeCollectionInput,
    resolveEffectiveCollectionId: resolveEffectiveCollectionId,
    sanitizeKnownCollections: sanitizeKnownCollections,
    isKnownCollection: isKnownCollection,
    formatCollectionDisplay: formatCollectionDisplay,
  };
});
