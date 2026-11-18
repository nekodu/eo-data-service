import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from eods import auth
from eods.cube import write_cube
from eods.synth import synthetic_dataset

ISSUER = "http://keycloak.test/realms/eods"
AUDIENCE = "eods-api"


@pytest.fixture(scope="session")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def make_token(rsa_key):
    def _make(scope="cube:read", aud=AUDIENCE, iss=ISSUER, exp_in=300, key=None):
        now = int(time.time())
        claims = {
            "sub": "svc-1",
            "azp": "test-client",
            "scope": scope,
            "aud": aud,
            "iss": iss,
            "iat": now,
            "exp": now + exp_in,
        }
        return jwt.encode(claims, key or rsa_key, algorithm="RS256", headers={"kid": "k1"})

    return _make


@pytest.fixture
def cube_dir(tmp_path):
    p = tmp_path / "cube.zarr"
    write_cube(synthetic_dataset(n_time=40), p)
    return p


@pytest.fixture
def client(cube_dir, tmp_path, rsa_key, monkeypatch):
    from fastapi.testclient import TestClient

    from eods import api, ml
    from eods.config import get_settings
    from eods.cube import open_cube

    ds = open_cube(cube_dir)
    clf, report = ml.train(
        ds["cloud"].values, ds.time.values, ml.NowcastConfig(samples_per_frame=500), True
    )
    model_path = tmp_path / "nowcast.joblib"
    ml.save(clf, report, model_path)

    monkeypatch.setenv("EODS_CUBE_PATH", str(cube_dir))
    monkeypatch.setenv("EODS_MODEL_PATH", str(model_path))
    monkeypatch.setenv("EODS_OIDC_ISSUER", ISSUER)
    get_settings.cache_clear()
    auth.set_verifier(auth.TokenVerifier(ISSUER, AUDIENCE, lambda _t: rsa_key.public_key()))
    with TestClient(api.app) as c:
        yield c
    auth.set_verifier(None)
    get_settings.cache_clear()
