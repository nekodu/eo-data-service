from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EODS_", env_file=".env", extra="ignore")

    cube_path: Path = Path("data/cube.zarr")
    model_path: Path = Path("models/nowcast.joblib")

    # OIDC: the API checks tokens itself even behind the gateway
    oidc_issuer: str = "http://localhost:8080/realms/eods"
    # in-cluster the JWKS is fetched from the service name, not the public issuer URL
    oidc_jwks_url: str | None = None
    oidc_audience: str = "eods-api"
    auth_disabled: bool = False

    max_cells: int = 250_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
