from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("eods_http_requests_total", "HTTP requests", ["route", "method", "status"])
LATENCY = Histogram(
    "eods_http_request_duration_seconds",
    "HTTP request latency",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
AUTH_FAILURES = Counter("eods_auth_failures_total", "Rejected requests by reason", ["reason"])
CELLS_SERVED = Counter("eods_cells_served_total", "Cube cells returned to clients", ["route"])
CUBE_LATEST = Gauge("eods_cube_latest_timestamp_seconds", "Time of the newest slot in the cube")
CUBE_STEPS = Gauge("eods_cube_time_steps", "Number of time steps in the cube")
MODEL_LOADED = Gauge("eods_model_loaded", "1 if the nowcast model is loaded")
