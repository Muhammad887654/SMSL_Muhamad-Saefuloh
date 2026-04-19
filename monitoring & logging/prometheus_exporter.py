"""
3.prometheus_exporter.py  ─  Custom Prometheus Exporter
=========================================================
Scrape statistik dari /stats endpoint Flask Corn Yield API,
lalu expose ke Prometheus via HTTP pada port 8000.

Metrik yang di-export
─────────────────────
  corn_api_up                    gauge    – 1=up, 0=down
  corn_api_requests_total        counter  – total request masuk
  corn_api_errors_total          counter  – total request error
  corn_api_avg_latency_ms        gauge    – rata-rata latency (ms)
  corn_api_error_rate            gauge    – rasio error/total (0–1)
  corn_api_avg_yield_kg          gauge    – rata-rata yield diprediksi (kg/acre)
  corn_api_yield_bucket_total    counter  – distribusi prediksi per rentang yield

Jalankan:  python 3.prometheus_exporter.py
Scrape:    http://localhost:8000/metrics
"""

import time, threading, logging, json
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    import requests as _req
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    HAS_REQUESTS = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("exporter.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("corn-exporter")

# ── Config ────────────────────────────────────────────────────────────────────
API_HEALTH = "http://localhost:5000/health"
API_STATS  = "http://localhost:5000/stats"
PORT       = 8000
INTERVAL   = 15   # detik

_cache = ""
_lock  = threading.Lock()


# ── Fetch JSON ────────────────────────────────────────────────────────────────
def fetch(url: str):
    try:
        if HAS_REQUESTS:
            r = _req.get(url, timeout=5)
            r.raise_for_status()
            return r.json()
        else:
            with urllib.request.urlopen(url, timeout=5) as r:
                return json.loads(r.read().decode())
    except Exception as e:
        logger.warning(f"fetch({url}) gagal: {e}")
        return None


# ── Collect ───────────────────────────────────────────────────────────────────
def collect() -> str:
    lines = []

    # ── 1. corn_api_up ───────────────────────────────────────────────────────
    health = fetch(API_HEALTH)
    up = 1 if (health and health.get("status") == "ok") else 0
    lines += [
        "# HELP corn_api_up Status Corn Yield API (1=up 0=down)",
        "# TYPE corn_api_up gauge",
        f"corn_api_up {up}",
    ]

    # API down → semua counter 0
    if not up:
        logger.warning("API down – set semua metrik ke 0")
        for name, typ, hlp in [
            ("corn_api_requests_total", "counter", "Total request"),
            ("corn_api_errors_total",   "counter", "Total error"),
            ("corn_api_avg_latency_ms", "gauge",   "Avg latency ms"),
            ("corn_api_error_rate",     "gauge",   "Error rate 0-1"),
            ("corn_api_avg_yield_kg",   "gauge",   "Avg yield kg/acre"),
        ]:
            lines += [f"# HELP {name} {hlp}", f"# TYPE {name} {typ}", f"{name} 0"]
        return "\n".join(lines) + "\n"

    # ── Fetch stats ───────────────────────────────────────────────────────────
    s = fetch(API_STATS)
    if not s:
        return "\n".join(lines) + "\n"

    req   = s.get("total_requests", 0)
    err   = s.get("total_errors",   0)
    lat   = s.get("avg_latency_ms", 0.0)
    rate  = s.get("error_rate",     0.0)
    yld   = s.get("avg_yield",      0.0)
    bkts  = s.get("yield_buckets",  {})

    # ── 2–6. Scalar metrics ───────────────────────────────────────────────────
    scalars = [
        ("corn_api_requests_total", "counter", "Total request ke Corn Yield API",         req),
        ("corn_api_errors_total",   "counter", "Total request yang menghasilkan error",    err),
        ("corn_api_avg_latency_ms", "gauge",   "Rata-rata waktu inferensi (ms)",           lat),
        ("corn_api_error_rate",     "gauge",   "Rasio error / total request (0–1)",        rate),
        ("corn_api_avg_yield_kg",   "gauge",   "Rata-rata prediksi yield jagung (kg/acre)",yld),
    ]
    for name, typ, hlp, val in scalars:
        lines += [f"# HELP {name} {hlp}", f"# TYPE {name} {typ}", f"{name} {val:.6g}"]

    # ── 7. Yield bucket distribution ──────────────────────────────────────────
    lines += [
        "# HELP corn_api_yield_bucket_total Distribusi prediksi per rentang yield",
        "# TYPE corn_api_yield_bucket_total counter",
    ]
    for rng, cnt in bkts.items():
        safe = rng.replace("-", "_").replace("+", "plus")
        lines.append(f'corn_api_yield_bucket_total{{range="{safe}"}} {cnt}')

    logger.info(
        f"Collected | up={up} req={req} err={err} "
        f"lat={lat:.1f}ms rate={rate:.4f} yield={yld:.1f}kg"
    )
    return "\n".join(lines) + "\n"


# ── Background thread ─────────────────────────────────────────────────────────
def scraper():
    global _cache
    while True:
        try:
            payload = collect()
            with _lock:
                _cache = payload
        except Exception as e:
            logger.error(f"Scrape error: {e}")
        time.sleep(INTERVAL)


# ── HTTP Handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            with _lock:
                body = _cache.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 – gunakan /metrics\n")

    def log_message(self, fmt, *args):
        logger.debug(f"HTTP {args}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Scrape awal …")
    with _lock:
        _cache = collect()

    t = threading.Thread(target=scraper, daemon=True)
    t.start()
    logger.info(f"Background scraper aktif setiap {INTERVAL}s")

    srv = HTTPServer(("0.0.0.0", PORT), Handler)
    logger.info(f"Prometheus Exporter → http://0.0.0.0:{PORT}/metrics")
    logger.info(f"Scraping dari       → {API_STATS}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        logger.info("Exporter dihentikan.")
        srv.server_close()
