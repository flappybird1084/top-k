# Top-K stable gateway

This Worker owns `https://top-k.andredlcruz.com`. It exposes only Top-K health,
API, and GitHub OAuth routes. `GPU_ORIGIN` is a secret containing the current
HTTPS compute origin. Keeping that value at the edge lets the public hostname
and GitHub callback remain stable when compute moves.

The Worker intentionally returns `enabled: false` from `/api/auth/config` until
a compute origin is connected. GitHub client credentials remain server-side;
never add them to this directory or the static website.

Deploy with `npx wrangler deploy`. Connect compute with
`npx wrangler secret put GPU_ORIGIN`, and remove it when the compute service is
offline. The Python gateway remains responsible for OAuth sessions and job
ownership, so GitHub OAuth credentials belong in that private compute
environment, not in the Worker.
