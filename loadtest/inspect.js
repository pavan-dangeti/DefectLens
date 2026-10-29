// Load test for POST /inspect.
//
// Run only once the coreset sweep has finished: other CPU work on the same machine skews latency.
//
//   k6 run -e BASE_URL=http://localhost:8000 -e CATEGORY=bottle \
//          -e IMAGE=loadtest/sample.png -e VUS=4 -e DURATION=2m loadtest/inspect.js
//
// A 503 (inference slots full) and a 429 (client over its rate limit) are the designed responses
// to overload, not failures: both are counted as refusals so the report shows how often the service
// refused work, and whether latency stayed bounded while it did. A single client IP hits the default
// limit of 60 images a minute almost at once, so a latency measurement needs the limit lifted on the
// server; the run then exercises inference, not the limiter.

import http from 'k6/http';
import { check } from 'k6';
import { Rate } from 'k6/metrics';

const image = open(__ENV.IMAGE || 'loadtest/sample.png', 'b');
const refused = new Rate('inference_refused');

export const options = {
  scenarios: {
    steady: {
      executor: 'constant-vus',
      vus: Number(__ENV.VUS || 4),
      duration: __ENV.DURATION || '2m',
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<1500'],
    inference_refused: ['rate<0.05'],
  },
};

http.setResponseCallback(http.expectedStatuses(200, 429, 503));

export default function () {
  const base = __ENV.BASE_URL || 'http://localhost:8000';
  const category = __ENV.CATEGORY || 'bottle';
  const response = http.post(`${base}/inspect?category=${category}`, {
    file: http.file(image, 'sample.png', 'image/png'),
  });
  refused.add(response.status === 429 || response.status === 503);
  check(response, {
    'status is 200, 429 or 503': (r) => [200, 429, 503].includes(r.status),
    'bounded decision or refusal': (r) =>
      r.status === 429 ||
      r.status === 503 ||
      r.json('decision') === 'good' ||
      r.json('decision') === 'anomaly',
  });
}
