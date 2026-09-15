const SITE_ORIGIN = "https://top-kernel-demo.andre520395.chatgpt.site";

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
  return origin === SITE_ORIGIN
    ? {
        "access-control-allow-origin": SITE_ORIGIN,
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

    if (!allowedPath(url.pathname)) return json({ error: "Not found" }, 404, cors);

    if (url.pathname.startsWith("/api/") && request.headers.get("origin") !== SITE_ORIGIN) {
      return json({ error: "Origin rejected" }, 403);
    }

    if (request.method === "OPTIONS") {
      if (request.headers.get("origin") !== SITE_ORIGIN) {
        return json({ error: "Origin rejected" }, 403);
      }
      return new Response(null, { status: 204, headers: cors });
    }

    if (url.pathname === "/health") {
      return json({
        edge: "ready",
        compute_origin_configured: Boolean(env.GPU_ORIGIN),
      });
    }

    if (url.pathname === "/api/auth/config" && !env.GPU_ORIGIN) {
      return json({ enabled: false }, 200, cors);
    }

    if (!env.GPU_ORIGIN) {
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
    headers.set("x-topk-edge", "cloudflare-worker");
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
