import {
  normalizeCollectionInput,
  resolveEffectiveCollectionId,
  sanitizeKnownCollections,
  isKnownCollection,
  formatCollectionDisplay,
  buildCrawlerPayload,
  normalizeCrawlerManualReview,
} from '../../static/theme/rag-tools-support.js';

describe('rag-tools-support helpers', () => {
  it('normalizes collection input by trimming whitespace', () => {
    expect(normalizeCollectionInput('  demo-collection  ')).toBe('demo-collection');
    expect(normalizeCollectionInput('\n\t scoped ')).toBe('scoped');
    expect(normalizeCollectionInput(undefined as unknown as string)).toBe('');
  });

  it('resolves effective collection id with legacy fallback when enabled', () => {
    expect(
      resolveEffectiveCollectionId({
        collectionInput: ' support-faq ',
        legacyDocClass: 'ignored',
        allowLegacy: true,
      })
    ).toBe('support-faq');
    expect(
      resolveEffectiveCollectionId({
        collectionInput: '  ',
        legacyDocClass: ' legacy-doc ',
        allowLegacy: true,
      })
    ).toBe('legacy-doc');
    expect(
      resolveEffectiveCollectionId({
        collectionInput: '  ',
        legacyDocClass: 'legacy-doc',
        allowLegacy: false,
      })
    ).toBe('');
  });

  it('sanitizes known collections and removes duplicates', () => {
    expect(sanitizeKnownCollections(['  onboarding  ', 'demo', 'onboarding', '', null as never])).toEqual([
      'onboarding',
      'demo',
    ]);
  });

  it('detects known collections in the sanitized list', () => {
    const known = sanitizeKnownCollections(['root', 'archive']);
    expect(isKnownCollection(' root ', known)).toBe(true);
    expect(isKnownCollection('ARCHIVE', known)).toBe(false);
    expect(isKnownCollection('', known)).toBe(false);
  });

  it('formats collection display with fallback placeholder', () => {
    expect(formatCollectionDisplay('primary')).toBe('primary');
    expect(formatCollectionDisplay('   ')).toBe('–');
  });

  it('builds crawler payload from loosely typed inputs', () => {
    const payload = buildCrawlerPayload({
      workflowId: ' wf-1 ',
      originUrl: ' https://example.com/page ',
      documentId: ' doc-1 ',
      provider: ' web ',
      title: ' Handbook ',
      language: ' DE ',
      content: '<html>Hallo</html>',
      contentType: ' text/html ',
      fetch: false,
      tags: 'alpha, beta , ',
      maxDocumentBytes: '1024',
      shadowMode: true,
      dryRun: false,
      snapshot: '1',
      snapshotLabel: ' debug ',
      manualReview: 'APPROVED',
      forceRetire: 1,
      recomputeDelta: '1',
    });

    expect(payload.workflow_id).toBe('wf-1');
    expect(payload.origin_url).toBe('https://example.com/page');
    expect(payload.document_id).toBe('doc-1');
    expect(payload.provider).toBe('web');
    expect(payload.title).toBe('Handbook');
    expect(payload.language).toBe('DE');
    expect(payload.content_type).toBe('text/html');
    expect(payload.content).toBe('<html>Hallo</html>');
    expect(payload.fetch).toBe(false);
    expect(payload.tags).toEqual(['alpha', 'beta']);
    expect(payload.max_document_bytes).toBe(1024);
    expect(payload.shadow_mode).toBe(true);
    expect(payload.dry_run).toBe(false);
    expect(payload.snapshot).toBe(true);
    expect(payload.snapshot_label).toBe('debug');
    expect(payload.manual_review).toBe('approved');
    expect(payload.force_retire).toBe(true);
    expect(payload.recompute_delta).toBe(true);
  });

  it('omits content when fetch is enabled and input empty', () => {
    const payload = buildCrawlerPayload({ fetch: true, content: '' });

    expect(payload.fetch).toBe(true);
    expect(Object.prototype.hasOwnProperty.call(payload, 'content')).toBe(false);
  });

  it('enforces empty content placeholder when fetch disabled', () => {
    const payload = buildCrawlerPayload({ fetch: false });

    expect(payload.fetch).toBe(false);
    expect(payload.content).toBe('');
  });

  it('normalizes crawler manual review inputs', () => {
    expect(normalizeCrawlerManualReview('required')).toBe('required');
    expect(normalizeCrawlerManualReview(' Approved ')).toBe('approved');
    expect(normalizeCrawlerManualReview('Rejected')).toBe('rejected');
    expect(normalizeCrawlerManualReview('')).toBeNull();
    expect(normalizeCrawlerManualReview('skip')).toBeNull();
  });
});
