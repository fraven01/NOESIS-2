import {
  normalizeCollectionInput,
  resolveEffectiveCollectionId,
  sanitizeKnownCollections,
  isKnownCollection,
  formatCollectionDisplay,
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
});
