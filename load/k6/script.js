import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';
import { textSummary } from './vendor/k6-summary-0.0.4.js';

const scopeLatency = new Trend('scope_latency_ms', true);
const scopeErrorRate = new Rate('scope_error_rate');

const baseUrl = (__ENV.STAGING_WEB_URL || '').replace(/\/$/, '');
const tenantSchema = __ENV.STAGING_TENANT_SCHEMA;
const tenantId = __ENV.STAGING_TENANT_ID;
const caseId = __ENV.STAGING_CASE_ID;
const bearerToken = __ENV.STAGING_BEARER_TOKEN;
const keyAlias = __ENV.STAGING_KEY_ALIAS;
const maxConcurrency = Number(__ENV.STAGING_WEB_CONCURRENCY || 30);
const spikeArrivalRate = Number(__ENV.SCOPE_SPIKE_RPS || maxConcurrency);
const soakArrivalRate = Number(
  __ENV.SCOPE_SOAK_RPS || Math.max(Math.floor(maxConcurrency / 2), 1),
);
const rampUpDuration = __ENV.SCOPE_RAMP_UP || '30s';
const soakDuration = __ENV.SCOPE_SOAK_DURATION || '2m';
const rampDownDuration = __ENV.SCOPE_RAMP_DOWN || '30s';
const p95ThresholdMs = Number(__ENV.SCOPE_P95_THRESHOLD_MS || 1200);
const thinkTimeSeconds = Number(__ENV.SCOPE_THINK_TIME_SECONDS || 0.5);
const idempotencyPrefix = __ENV.SCOPE_IDEMPOTENCY_PREFIX || 'k6-chaos-load';
const defaultPrompt = JSON.stringify(
  __ENV.SCOPE_REQUEST_BODY
    ? JSON.parse(__ENV.SCOPE_REQUEST_BODY)
    : {
        prompt: 'Summarise the latest release notes for internal QA.',
        metadata: { origin: 'k6-spike-soak', environment: 'staging' },
      },
);

if (!baseUrl) {
  throw new Error(
    'STAGING_WEB_URL is required. Refer to docs/cloud/gcp-staging.md for the Cloud Run endpoint.',
  );
}
if (!tenantSchema || !tenantId || !caseId) {
  throw new Error(
    'STAGING_TENANT_SCHEMA, STAGING_TENANT_ID, and STAGING_CASE_ID must be provided (see docs/api/reference.md).',
  );
}

export const options = {
  scenarios: {
    'spike-and-soak-short': {
      executor: 'ramping-arrival-rate',
      startRate: Math.max(Math.floor(spikeArrivalRate / 3), 1),
      timeUnit: '1s',
      preAllocatedVUs: Math.max(spikeArrivalRate * 2, maxConcurrency * 2),
      maxVUs: Math.max(spikeArrivalRate * 3, maxConcurrency * 3),
      stages: [
        { target: spikeArrivalRate, duration: rampUpDuration },
        { target: soakArrivalRate, duration: soakDuration },
        { target: 0, duration: rampDownDuration },
      ],
    },
  },
  thresholds: {
    'http_req_failed{scenario:spike-and-soak-short}': ['rate<0.05'],
    'http_req_duration{scenario:spike-and-soak-short}': [`p(95)<${p95ThresholdMs}`],
    scope_error_rate: ['rate<0.05'],
  },
  discardResponseBodies: false,
};

function buildHeaders(iterationId) {
  const headers = {
    'Content-Type': 'application/json',
    'X-Tenant-Schema': tenantSchema,
    'X-Tenant-ID': tenantId,
    'X-Case-ID': caseId,
    'Idempotency-Key': `${idempotencyPrefix}-${iterationId}`,
  };

  if (bearerToken) {
    headers.Authorization = bearerToken.startsWith('Bearer ')
      ? bearerToken
      : `Bearer ${bearerToken}`;
  }
  if (keyAlias) {
    headers['X-Key-Alias'] = keyAlias;
  }

  return headers;
}

export default function spikeAndSoak() {
  const iterationId = `${__VU}-${__ITER}-${Date.now()}`;
  const res = http.post(`${baseUrl}/ai/scope/`, defaultPrompt, {
    headers: buildHeaders(iterationId),
    tags: { tenant: tenantId, scenario: 'spike-and-soak-short' },
  });

  scopeLatency.add(res.timings.duration);
  const failed = res.status >= 400;
  scopeErrorRate.add(failed);

  check(res, {
    'status is success or accepted': (r) => r.status === 200 || r.status === 202,
    'has trace id': (r) => Boolean(r.headers['X-Trace-Id'] || r.json('trace_id')),
  });

  if (thinkTimeSeconds > 0) {
    sleep(thinkTimeSeconds);
  }
}

export function handleSummary(data) {
  const scopeMetrics = data.metrics.scope_latency_ms || {};
  const percentiles = scopeMetrics && scopeMetrics.percentiles ? scopeMetrics.percentiles : {};

  return {
    'k6-summary.json': JSON.stringify(data, null, 2),
    'k6-summary.txt': textSummary(data, { indent: ' ', enableColors: false }),
    'scope-latency.json': JSON.stringify(
      {
        p95_threshold_ms: p95ThresholdMs,
        percentiles,
        thresholds: options.thresholds,
      },
      null,
      2,
    ),
  };
}
