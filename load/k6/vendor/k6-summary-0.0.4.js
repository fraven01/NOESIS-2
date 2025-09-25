// Vendored from https://jslib.k6.io/k6-summary/0.0.4/index.js
// The original file is licensed under the Apache-2.0 license.
// We keep a simplified but compatible implementation to avoid remote imports in CI.

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'n/a';
  }
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.01) {
    return value.toExponential(2);
  }
  return value.toFixed(2);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'n/a';
  }
  return `${(value * 100).toFixed(2)}%`;
}

function color(text, ansi, enable) {
  if (!enable) {
    return text;
  }
  return `${ansi}${text}\u001b[0m`;
}

function renderThreshold(name, threshold, indent, enableColors) {
  const status = threshold.ok ? color('PASS', '\u001b[32m', enableColors) : color('FAIL', '\u001b[31m', enableColors);
  const value = threshold.actual !== undefined ? threshold.actual : threshold.value;
  const formatted = typeof value === 'number' ? formatNumber(value) : `${value}`;
  return `${indent}- ${status} ${name} (${formatted})`;
}

function renderMetric(name, metric, indent, enableColors) {
  const lines = [`${indent}${name}:`];
  const stats = [];

  if (typeof metric.count === 'number') {
    stats.push(`count=${formatNumber(metric.count)}`);
  }
  if (typeof metric.rate === 'number') {
    stats.push(`rate=${formatNumber(metric.rate)}`);
  }
  if (typeof metric.min === 'number') {
    stats.push(`min=${formatNumber(metric.min)}`);
  }
  if (typeof metric.max === 'number') {
    stats.push(`max=${formatNumber(metric.max)}`);
  }
  if (typeof metric.avg === 'number') {
    stats.push(`avg=${formatNumber(metric.avg)}`);
  }
  if (typeof metric.med === 'number') {
    stats.push(`med=${formatNumber(metric.med)}`);
  }
  if (typeof metric.p95 === 'number') {
    stats.push(`p95=${formatNumber(metric.p95)}`);
  }
  if (typeof metric.p99 === 'number') {
    stats.push(`p99=${formatNumber(metric.p99)}`);
  }
  if (typeof metric.total === 'number') {
    stats.push(`total=${formatNumber(metric.total)}`);
  }

  if (stats.length) {
    lines.push(`${indent}  ${stats.join(', ')}`);
  }

  if (metric.percentiles && typeof metric.percentiles === 'object') {
    const pct = Object.entries(metric.percentiles)
      .map(([key, value]) => `${key}=${formatNumber(value)}`)
      .join(', ');
    if (pct) {
      lines.push(`${indent}  percentiles: ${pct}`);
    }
  }

  if (metric.thresholds) {
    const thresholdLines = [];
    for (const [thresholdName, threshold] of Object.entries(metric.thresholds)) {
      thresholdLines.push(renderThreshold(thresholdName, threshold, `${indent}  `, enableColors));
    }
    if (thresholdLines.length) {
      lines.push(`${indent}  thresholds:`);
      lines.push(...thresholdLines);
    }
  }

  return lines.join('\n');
}

export function textSummary(data, options = {}) {
  const { indent = '', enableColors = true } = options;
  const lines = [];

  lines.push(`${indent}execution summary`);
  lines.push(`${indent}${'='.repeat('execution summary'.length)}`);

  if (data.state) {
    lines.push(`${indent}state: ${data.state}`);
  }
  if (data.testRunDurationMs) {
    lines.push(`${indent}duration: ${formatNumber(data.testRunDurationMs / 1000)}s`);
  }

  if (data.metrics) {
    const metricNames = Object.keys(data.metrics).sort();
    for (const name of metricNames) {
      const metric = data.metrics[name];
      if (!metric) continue;
      lines.push(renderMetric(name, metric, `${indent}`, enableColors));
    }
  }

  if (data.root_group && data.root_group.checks) {
    const checks = data.root_group.checks;
    const checkPasses = checks.passes || 0;
    const checkFails = checks.fails || 0;
    const totalChecks = checkPasses + checkFails;
    const passRate = totalChecks === 0 ? 0 : checkPasses / totalChecks;
    lines.push(`${indent}checks:`);
    lines.push(
      `${indent}  passes=${checkPasses} fails=${checkFails} pass_rate=${formatPercent(passRate)}`,
    );
  }

  return `${lines.join('\n')}\n`;
}

export default { textSummary };
