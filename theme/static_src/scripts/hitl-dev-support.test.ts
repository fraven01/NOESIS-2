import {
  clampReason,
  splitCustomUrls,
  validateCustomUrls,
  formatCountdown,
  computeDecisionSets,
  normalizeFacetScores,
} from '../../static/theme/hitl-dev-support.js';

describe('hitl-dev-support helpers', () => {
  it('clamps reasons to 280 characters without stripping whitespace entirely', () => {
    const long = 'a'.repeat(500);
    expect(clampReason(long)).toHaveLength(280);
    expect(clampReason('   reason  ')).toBe('reason');
    expect(clampReason(undefined as unknown as string)).toBe('');
  });

  it('splits and validates custom URLs', () => {
    const urls = splitCustomUrls(' https://example.com/a  \nhttp://test.local/path\ninvalid\n');
    expect(urls).toEqual(['https://example.com/a', 'http://test.local/path', 'invalid']);

    const validation = validateCustomUrls(urls);
    expect(validation.valid).toEqual(['https://example.com/a', 'http://test.local/path']);
    expect(validation.invalid).toEqual(['invalid']);
  });

  it('formats countdown labels for upcoming and overdue deadlines', () => {
    const reference = new Date('2024-01-01T12:00:00Z');
    const upcoming = formatCountdown('2024-01-01T12:05:30Z', reference);
    expect(upcoming.overdue).toBe(false);
    expect(upcoming.label).toBe('Auto-Approve in 05:30');

    const overdue = formatCountdown('2024-01-01T11:59:00Z', reference);
    expect(overdue.overdue).toBe(true);
    expect(overdue.label).toBe('Auto-approved');
  });

  it('computes decision sets for different actions', () => {
    const ids = ['a', 'b', 'c'];
    expect(computeDecisionSets(ids, ['a'], 'approve_all')).toEqual({ approved: ids, rejected: [] });
    expect(computeDecisionSets(ids, ['a'], 'reject_all')).toEqual({ approved: [], rejected: ids });
    expect(computeDecisionSets(ids, ['a', 'b'], 'approve_selected')).toEqual({
      approved: ['a', 'b'],
      rejected: ['c'],
    });
    expect(computeDecisionSets(ids, ['a', 'b'], 'reject_selected')).toEqual({
      approved: ['c'],
      rejected: ['a', 'b'],
    });
  });

  it('normalizes facet scores to two decimals', () => {
    const normalized = normalizeFacetScores({ TECHNICAL: 0.71234, INVALID: 'nope' as never });
    expect(normalized).toEqual({ TECHNICAL: 0.71 });
  });
});
