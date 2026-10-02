# GitHub sign-in

Website: https://top-k.dev/

The hosted EC2 backend has been removed. The website is a static frontend:
it makes no gateway API calls and offers no hosted sign-in or search. Runs
start through the CLI. The gateway configuration below is retained as
reference for the retired deployment.

The landing page is public and provides CLI instructions. The hosted dashboard
and search pages are unpublished. The private gateway admits only
`andred1729` and `flappybird1084` by default. Set
`JUDGES_ALLOWED_GITHUB_LOGINS=andred1729,flappybird1084` in production;
the literal `*` is not a wildcard. Completed public repository names may
appear on the landing orbit, but owner identities, run IDs, datasets,
metrics, and traces remain private.

The CLI runs on the user's own GPU and logs to their own W&B account. The
private API still enforces per-user notebook and W&B credentials for its two
approved accounts; it never hands out operator notebooks or keys.

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
   - Authorization callback URL: https://api.top-k.dev/auth/github/callback
3. Set these in the gateway's private environment only (see `.env.example`):
   `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`,
   `GITHUB_AUTH_ORIGIN=https://api.top-k.dev`, `GITHUB_AUTH_DB`,
   `JUDGES_EXPIRES_AT`, `JUDGES_INTEGRATION_DIR`, and a freshly generated
   `JUDGES_EDGE_SECRET` (`python -c "import secrets;print(secrets.token_urlsafe(48))"`).
4. Give the Worker the same secret and the compute origin:
   `npx wrangler secret put EDGE_SECRET` and `npx wrangler secret put GPU_ORIGIN`.
   Remove `GPU_ORIGIN` when compute is offline. Neither value belongs in
   `wrangler.jsonc`, the repository, or the static site.
5. Publish the CLI-only landing page (`ui/`) to the Sites project. Do not
   publish the removed dashboard, search, or recorded notebook pages. Private
   Weave links and operator run logs must not be exposed as static assets.
6. Test sign-in, sign-out, callback replay rejection, and ownership with two
   distinct GitHub accounts before activating. Unit tests mock GitHub and
   cannot verify a real app registration.

## What a visitor connects

For the private gateway API, a visitor creates or signs in to marimo,
starts a notebook on a GPU runtime, chooses **Pair with agent**, supplies that
prompt, then adds a W&B API key. Both are stored server-side under `JUDGES_INTEGRATION_DIR`,
in files named by a hash of the GitHub user id, created with owner-only
permissions from the first byte, in a directory only the gateway user can
enter. Neither the notebook token nor the W&B key is ever returned by the API,
written into a job's public snapshot or job file, or stored in the browser —
responses only say whether each integration is connected. A run reads them at
launch, straight into the child process environment.

Only marimo's own notebook hosting is accepted (`*.molab.run`, `*.marimo.io`,
`*.marimo.app`; override with `JUDGES_NOTEBOOK_HOSTS`). The gateway connects to
whatever address is pasted here, so accepting an arbitrary one would make this
server a request forwarder into its own network. Credentials in the URL, a
non-standard port, a query string, a fragment, and an odd path are all refused.

Runs execute concurrently — each visitor is on their own GPU, so there is
nothing to serialize. `JUDGES_CONCURRENT_RUNS` (default 4) bounds how many run
at once, and one account may hold only one of those slots at a time.

Each run reports to its owner's W&B account and to no other. A visitor who
connected no W&B key gets no mirroring at all; the server's own
`WANDB_API_KEY` is never used for a public run.

## Sessions and identity

Sign-in uses state, PKCE, a Secure/HttpOnly/SameSite=Lax callback cookie, and
one-use stored states. The GitHub access token validates `/user` server-side and
is then discarded; the browser gets an opaque application session instead, and
only its hash is stored. A session idles out after an hour of no activity;
authenticated API calls slide that window forward, so watching a long run does
not sign you out, and it cannot be extended past six hours in total. A new
token is minted only by a fresh GitHub sign-in — checking who is signed in
never replaces the token, so several open tabs cannot revoke each other. Sign-out revokes immediately, and a sign-out
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

## What a notebook may ask this server to do

A run executing on a visitor's notebook can write request files into a relay
directory that the dispatcher on this machine services. That is how a run
reaches things the notebook cannot: the operator's Claude/Codex subscription
sessions and a private search service. Whoever controls the notebook controls
what lands in that directory, so the boundary is explicit:

* **Closed by default for a visitor's run.** A job owned by a signed-in visitor
  gets no operator credential at all unless the operator sets
  `KEVO_ALLOW_OPERATOR_LLM_RELAY=1` (and `KEVO_ALLOW_OPERATOR_SEARCH_RELAY=1`
  for search). Left unset — the default — those runs must supply their own API
  key. This is deliberate: nothing can prove a request file was written by the
  harness rather than by the notebook's owner, so the answer to an
  unprovable-authorship request is no.
* **Bound to the active run.** Every request must carry the secret stamped into
  the launch environment of the run currently being dispatched. Anything
  without it, and anything arriving after that run exits, is refused.
* **Budgeted.** Per run: how many requests, how many tokens, how large one
  prompt may be, how many searches, how long a query. Per account, across runs,
  inside a rolling window: `KEVO_RELAY_OWNER_MAX_REQUESTS`,
  `KEVO_RELAY_OWNER_MAX_TOKENS`, `KEVO_RELAY_OWNER_MAX_SEARCHES`. Crossing
  either closes the relay for the rest of that run.
* **Restricted.** Only the model names in `KEVO_RELAY_MODELS` (a sensible
  default list) may be requested, and a request must be a well-formed list of
  system/user/assistant text messages.
* **Answered once.** A request id that has been answered, refused, expired or
  dropped is never picked up again, so a request the sandbox has not collected
  cannot be run — or billed — a second time.

The relay directory on the notebook is root-owned and sticky (mode 01733): the
run's own Unix identity can drop a request in and read the answer addressed to
it, but cannot list the directory, nor delete or replace files it does not own.
