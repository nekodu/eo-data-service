"""Bearer-token checks against the Keycloak realm.

The gateway already rejects unsigned or expired tokens, but the API checks again and
is the only place that enforces scopes, so a misrouted request can't skip authorisation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, Request, status

from .config import Settings, get_settings
from .metrics import AUTH_FAILURES

ALGORITHMS = ["RS256"]


@dataclass(frozen=True)
class Principal:
    subject: str
    client_id: str | None
    scopes: frozenset[str]


class TokenVerifier:
    def __init__(self, issuer: str, audience: str, key_resolver: Callable[[str], object]):
        self.issuer = issuer
        self.audience = audience
        self._key_for = key_resolver

    @classmethod
    def from_jwks(cls, issuer: str, audience: str, jwks_url: str) -> TokenVerifier:
        client = jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=300, timeout=5)
        return cls(issuer, audience, lambda token: client.get_signing_key_from_jwt(token).key)

    def verify(self, token: str) -> Principal:
        key = self._key_for(token)
        claims = jwt.decode(
            token,
            key=key,
            algorithms=ALGORITHMS,
            audience=self.audience,
            issuer=self.issuer,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            leeway=10,
        )
        scopes = frozenset(str(claims.get("scope", "")).split())
        return Principal(subject=claims["sub"], client_id=claims.get("azp"), scopes=scopes)


_verifier: TokenVerifier | None = None


def get_verifier(settings: Settings = Depends(get_settings)) -> TokenVerifier:
    global _verifier
    if _verifier is None:
        jwks = settings.oidc_jwks_url or f"{settings.oidc_issuer}/protocol/openid-connect/certs"
        _verifier = TokenVerifier.from_jwks(settings.oidc_issuer, settings.oidc_audience, jwks)
    return _verifier


def set_verifier(v: TokenVerifier | None) -> None:
    global _verifier
    _verifier = v


def _deny(code: int, reason: str, detail: str) -> HTTPException:
    AUTH_FAILURES.labels(reason=reason).inc()
    headers = {"WWW-Authenticate": "Bearer"} if code == status.HTTP_401_UNAUTHORIZED else None
    return HTTPException(status_code=code, detail=detail, headers=headers)


def require_scope(scope: str):
    def dep(
        request: Request,
        settings: Settings = Depends(get_settings),
        verifier: TokenVerifier = Depends(get_verifier),
    ) -> Principal:
        if settings.auth_disabled:
            return Principal("anonymous", None, frozenset({scope}))
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            raise _deny(401, "missing_token", "missing bearer token")
        try:
            principal = verifier.verify(header[7:].strip())
        except jwt.ExpiredSignatureError:
            raise _deny(401, "expired", "token expired") from None
        except (jwt.InvalidAudienceError, jwt.InvalidIssuerError):
            raise _deny(401, "wrong_audience_or_issuer", "token not issued for this API") from None
        except (jwt.PyJWTError, jwt.PyJWKClientError):
            raise _deny(401, "invalid", "invalid token") from None
        if scope not in principal.scopes:
            raise _deny(403, "missing_scope", f"scope '{scope}' required")
        request.state.client_id = principal.client_id
        return principal

    return dep
