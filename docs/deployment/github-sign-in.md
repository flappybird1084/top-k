# GitHub sign-in

Website: https://top-kernel-demo.andre520395.chatgpt.site/?demo=1

GitHub identity is handled by the existing Python gateway, not by the static Sites host or its ChatGPT sign-in system. The website address stays unchanged. This integration is prepared but requires OAuth registration and a stable gateway deployment before publishing the new client.

1. Deploy `judges_server.py` behind a stable HTTPS origin. Do not register an expiring `trycloudflare.com` address for permanent OAuth.
2. Create a GitHub OAuth App at https://github.com/settings/applications/new:
   - Application name: Top-Kernel
   - Homepage URL: https://top-kernel-demo.andre520395.chatgpt.site/
   - Authorization callback URL: your gateway origin followed by `/auth/github/callback`.
3. Set these values only in the gateway's private environment:
   - `GITHUB_CLIENT_ID`
   - `GITHUB_CLIENT_SECRET`
   - `GITHUB_AUTH_ORIGIN`: the fixed HTTPS gateway origin, without a trailing path
   - `GITHUB_AUTH_DB`: durable private SQLite path (defaults to `~/.local/state/kernel-evolution/github-auth.sqlite`)
4. Replace the `backend` origin in `ui/live-api.js` with that gateway origin. Copy the updated client, `github-auth.css`, and `front.js` into the existing Sites checkout's `dist` directory. Publish the same Sites project; do not create another site or change its slug.
5. Test sign-in, sign-out, callback replay rejection, and ownership with two distinct GitHub accounts before merging and activating. Unit tests mock GitHub and cannot verify a real app registration.

The gateway uses state, PKCE, a Secure/HttpOnly/SameSite=Lax callback cookie, and one-use stored states. GitHub access tokens are used server-side to validate `/user`, then discarded. The client receives a six-hour opaque application session; only its hash is persisted. Sign-out revokes it. Sessions identify users by immutable GitHub numeric ID, not their changeable login name. Legacy anonymous bearer tokens are rejected when GitHub auth is configured; old anonymous runs are not automatically assigned to an account.

The flow requests `read:user`, not private repository access. Signing in does not enable cloning private repositories. Frontend session storage and the static site's origin remain security-sensitive; keep scripts trusted and do not place provider secrets in static files.

The original judging service had a five-hour expiration. Authentication remains available after that expiration, but launching GPU runs still returns the original closed-window response. Enabling OAuth does not extend the GPU window, restart jobs, or provision compute.
