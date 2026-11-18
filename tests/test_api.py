from cryptography.hazmat.primitives.asymmetric import rsa


def bearer(t):
    return {"Authorization": f"Bearer {t}"}


def test_health_needs_no_token(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").json()["model"] is True


def test_missing_token_is_401(client):
    r = client.get("/v1/cube")
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_cube_info_flags_synthetic(client, make_token):
    r = client.get("/v1/cube", headers=bearer(make_token()))
    assert r.status_code == 200
    body = r.json()
    assert body["synthetic"] is True
    assert body["shape"]["time"] == 40


def test_wrong_audience_rejected(client, make_token):
    assert client.get("/v1/cube", headers=bearer(make_token(aud="other-api"))).status_code == 401


def test_wrong_issuer_rejected(client, make_token):
    assert (
        client.get(
            "/v1/cube", headers=bearer(make_token(iss="http://evil/realms/eods"))
        ).status_code
        == 401
    )


def test_expired_rejected(client, make_token):
    assert client.get("/v1/cube", headers=bearer(make_token(exp_in=-60))).status_code == 401


def test_foreign_signature_rejected(client, make_token):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert client.get("/v1/cube", headers=bearer(make_token(key=other))).status_code == 401


def test_alg_none_rejected(client):
    import jwt

    t = jwt.encode({"sub": "x", "scope": "cube:read"}, key=None, algorithm="none")
    assert client.get("/v1/cube", headers=bearer(t)).status_code == 401


def test_nowcast_needs_its_own_scope(client, make_token):
    body = {"y1": 64, "x1": 64}
    assert (
        client.post("/v1/nowcast", json=body, headers=bearer(make_token("cube:read"))).status_code
        == 403
    )
    r = client.post("/v1/nowcast", json=body, headers=bearer(make_token("cube:read nowcast:run")))
    assert r.status_code == 200
    out = r.json()
    assert out["lead_minutes"] == 60
    assert 0 <= out["cloud_fraction_forecast"] <= 1


def test_nowcast_window_outside_cube_is_422(client, make_token):
    h = bearer(make_token("cube:read nowcast:run"))
    assert client.post("/v1/nowcast", json={"y1": 5000, "x1": 4}, headers=h).status_code == 422


def test_cloud_fraction_series(client, make_token):
    r = client.get(
        "/v1/cube/cloud-fraction", params={"y1": 50, "x1": 50}, headers=bearer(make_token())
    )
    assert r.status_code == 200
    series = r.json()["series"]
    assert len(series) == 40
    assert all(0 <= s["cloud_fraction"] <= 1 for s in series)


def test_window_limits(client, make_token, monkeypatch):
    h = bearer(make_token())
    assert (
        client.get(
            "/v1/cube/window", params={"time": "2026-01-01T01:00", "y1": 10, "x1": 10}, headers=h
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/v1/cube/window", params={"time": "2026-01-01T01:00", "y1": 999, "x1": 10}, headers=h
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/v1/cube/window",
            params={"time": "2026-01-01T01:00", "y0": 5, "y1": 5, "x1": 10},
            headers=h,
        ).status_code
        == 422
    )


def test_metrics_exposed(client, make_token):
    client.get("/v1/cube", headers=bearer(make_token()))
    client.get("/v1/cube")
    text = client.get("/metrics").text
    assert 'eods_http_requests_total{method="GET",route="/v1/cube",status="200"}' in text
    assert 'eods_auth_failures_total{reason="missing_token"}' in text
