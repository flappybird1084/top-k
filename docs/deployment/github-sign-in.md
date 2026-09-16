# GitHub sign-in

Website: https://top-kernel-demo.andre520395.chatgpt.site/

Everything on the site is behind GitHub sign-in: a visitor who is not signed in
sees only the sign-in card. Any signed-in GitHub user may submit any public
repository — that open access is intentional, and there is no allowlist.

After signing in, each visitor connects **their own** GPU notebook and **their
own** Weights & Biases account before a run can start. The operator's notebooks
and API keys are never handed to a visitor, and a run that has no notebook of
its own is refused (HTTP 428) rather than falling back to ours.

## The pieces

| Piece | Role |
| --- | --- |
| Cloudflare Worker (`cloudflare/gateway-worker`) | Owns the stable public hostname. Forwards only health, `/api/`, and the two OAuth routes. Adds the edge secret and the client's address. |
| `judges_server.py` | The origin. Verifies the edge secret, requires a signed-in GitHub user, owns rate limits, per-user integrations, and job ownership. |
| `github_signin.py` | OAuth with state + PKCE, and short rotated browser sessions. |

Both halves fail closed. The Worker refuses to proxy unless it has both a
compute origin and an edge secret; the gateway refuses every request unless it
has the edge secret and sees it on the request.

## Deployment

1. Deploy `judges_server.py` behind a stable HTTPS origin. Do not register an
   expiring `trycloudflare.com` address for permanent OAuth.
2. Create a GitHub OAuth App at https://github.com/settings/applications/new:
   - Application name: Top-Kernel
   - Homepage URL: https://top-kernel-demo.andre520395.chatgpt.site/
   - Authorization callback URL: https://top-k.andredlcruz.com/auth/github/callback
3. Set these in the gateway's private environment only (see `.env.example`):
   `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`,
   `GITHUB_AUTH_ORIGIN=https://top-k.andredlcruz.com`, `GITHUB_AUTH_DB`,
   `JUDGES_EXPIRES_AT`, `JUDGES_INTEGRATION_DIR`, and a freshly generated
   `JUDGES_EDGE_SECRET` (`python -c "import secrets;print(secrets.token_urlsafe(48))"`).
4. Give the Worker the same secret and the compute origin:
   `npx wrangler secret put EDGE_SECRET` and `npx wrangler secret put GPU_ORIGIN`.
   Remove `GPU_ORIGIN` when compute is offline. Neither value belongs in
   `wrangler.jsonc`, the repository, or the static site.
5. Publish the static client (`ui/`) to the Sites project as before. **Do not
   publish `ui/assets/recorded/**` or `ui/assets/recorded-data.js`**: the
   recorded demo evidence contains private Weave links and run logs from the
   operator's W&B account. The signed-in client fetches it from
   `/api/recorded/runs` instead.
6. Test sign-in, sign-out, callback replay rejection, and ownership with two
   distinct GitHub accounts before activating. Unit tests mock GitHub and
   cannot verify a real app registration.

## What a visitor connects

The setup card after sign-in walks through it: create or sign in to marimo,
start a notebook on a GPU runtime, choose **Pair with agent**, paste that whole
prompt (the token on screen is masked — only the copied text carries it), then
add a W&B API key. Both are stored server-side under `JUDGES_INTEGRATION_DIR`,
in files named by a hash of the GitHub user id and readable only by the gateway
user. Neither the notebook token nor the W&B key is ever returned by the API,
written into a job's public snapshot, or stored in the browser — responses only
say whether each integration is connected.

## Sessions and identity

Sign-in uses state, PKCE, a Secure/HttpOnly/SameSite=Lax callback cookie, and
one-use stored states. The GitHub access token validates `/user` server-side and
is then discarded; the browser gets an opaque application session instead, and
only its hash is stored. Sessions idle out after an hour, rotate on every
identity check (the presented token is revoked as the new one is issued), and
cannot be extended past six hours. Sign-out revokes immediately, and a sign-out
call without a valid session is refused rather than silently accepted.

Users are identified by the immutable GitHub numeric id, not the changeable
login name, and every run is scoped to that id — one account cannot read
another's runs, and the legacy anonymous-session endpoint is gone.

Sign-in requests, identity checks, and run creation are each rate limited, with
a hard cap on how many client addresses the limiter tracks so the limiter
itself cannot be used to exhaust memory. The client address comes from the
Worker's `X-TopK-Client-IP` header, which is only believed because the request
also carried the edge secret; the Worker strips any client-supplied copy of
both headers.

`/api/health` reports readiness and nothing else — no queue depth, worker
count, or exact expiry.

The OAuth flow requests `read:user`, not repository access, so signing in does
not enable cloning private repositories.

The original judging service had a five-hour expiration. Authentication stays
available afterwards, but launching GPU runs still returns the closed-window
response. Enabling OAuth does not extend the GPU window or provision compute.
