# Top-K stable gateway

This Worker owns the backend at `https://api.top-k.dev` and keeps
`https://top-k.andredlcruz.com` as a compatibility alias. The project site is
served directly at `https://top-k.dev` by Sites. The API domain exposes Top-K
health, API, and GitHub OAuth routes. Keeping compute at the edge lets the
backend and GitHub callback stay stable when compute moves.

Two secrets, both server-side:

- `GPU_ORIGIN` — the current HTTPS compute origin.
- `EDGE_SECRET` — shared with the Python gateway, sent as `X-TopK-Edge-Auth` so
  the origin can tell an edge request from a direct one. The Worker also strips
  any client-supplied `X-TopK-Edge-Auth`/`X-TopK-Client-IP` before setting its
  own, so the client address the origin rate-limits on cannot be spoofed.

Missing either one is a half-configured deployment: `/health` reports
`ready: false`, `/api/auth/config` reports `enabled: false` so the site can say
so plainly, and every other route returns 503 rather than proxying.

Deploy with `npx wrangler deploy`. Set the secrets with
`npx wrangler secret put GPU_ORIGIN` and `npx wrangler secret put EDGE_SECRET`,
and remove `GPU_ORIGIN` when the compute service is offline. GitHub OAuth
credentials and per-user integrations belong to the Python gateway's private
environment, never to this directory or the static website.
