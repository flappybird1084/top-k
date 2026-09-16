const SITE_ORIGIN = "https://top-kernel-demo.andre520395.chatgpt.site";
const SITE_ORIGINS = new Set([SITE_ORIGIN, "https://top-k.dev"]);

function json(body, status = 200, extra = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      "x-content-type-options": "nosniff",
      ...extra,
    },
  });
}

function corsHeaders(request) {
  const origin = request.headers.get("origin");
  return SITE_ORIGINS.has(origin)
    ? {
        "access-control-allow-origin": origin,
        "access-control-allow-methods": "GET, POST, HEAD, OPTIONS",
        "access-control-allow-headers": "Authorization, Content-Type, Idempotency-Key",
        vary: "Origin",
      }
    : {};
}

function allowedPath(pathname) {
  return (
    pathname === "/health" ||
    pathname.startsWith("/api/") ||
    pathname === "/auth/github/login" ||
    pathname === "/auth/github/callback"
  );
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const cors = corsHeaders(request);

    // The compatibility gateway hostname is human-facing too. Keep the
    // submitted Sites URL canonical instead of showing an API 404 at its root.
    if (!allowedPath(url.pathname)) {
      const destination = new URL(url.pathname + url.search, SITE_ORIGIN);
      return Response.redirect(destination.toString(), 302);
    }

    if (url.pathname.startsWith("/api/") && !SITE_ORIGINS.has(request.headers.get("origin"))) {
      return json({ error: "Origin rejected" }, 403);
    }

    if (request.method === "OPTIONS") {
      if (!SITE_ORIGINS.has(request.headers.get("origin"))) {
        return json({ error: "Origin rejected" }, 403);
      }
      return new Response(null, { status: 204, headers: cors });
    }

    // Readiness only. Which half of the configuration is missing, and whether
    // compute is attached, is operational detail a prober does not get.
    const ready = Boolean(env.GPU_ORIGIN && env.EDGE_SECRET);
    if (url.pathname === "/health") return json({ ready });

    if (url.pathname === "/api/auth/config" && !ready) {
      return json({ enabled: false }, 200, cors);
    }

    if (!ready) {
      return json({ error: "The compute service is not connected." }, 503, cors);
    }

    const targetBase = new URL(env.GPU_ORIGIN);
    if (targetBase.protocol !== "https:" || targetBase.username || targetBase.password) {
      return json({ error: "The compute service is misconfigured." }, 503, cors);
    }

    const target = new URL(url.pathname + url.search, targetBase);
    const headers = new Headers(request.headers);
    headers.delete("host");
    headers.delete("cf-connecting-ip");
    headers.delete("x-forwarded-for");
    headers.delete("x-topk-edge");
    // Drop any client-supplied copy before setting our own: the origin trusts
    // these two headers, so they must come from here and nowhere else.
    headers.delete("x-topk-edge-auth");
    headers.delete("x-topk-client-ip");
    headers.set("x-topk-edge-auth", env.EDGE_SECRET);
    const clientIp = request.headers.get("cf-connecting-ip");
    if (clientIp) headers.set("x-topk-client-ip", clientIp);
    const upstream = await fetch(target, {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
      redirect: "manual",
    });
    const responseHeaders = new Headers(upstream.headers);
    for (const [key, value] of Object.entries(cors)) responseHeaders.set(key, value);
    responseHeaders.set("cache-control", "no-store");
    responseHeaders.set("x-content-type-options", "nosniff");
    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders,
    });
  },
};
