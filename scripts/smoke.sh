#!/usr/bin/env bash
# End-to-end checks through the gateway: token issuance, authn, authz, network policy.
set -uo pipefail
cd "$(dirname "$0")/.."
SECRETS=deploy/.secrets
SVC=$(kubectl -n envoy-gateway-system get svc -l gateway.envoyproxy.io/owning-gateway-name=eods \
  -o jsonpath='{.items[0].metadata.name}')
kubectl -n envoy-gateway-system port-forward "svc/$SVC" 18080:80 >/dev/null 2>&1 &
PF=$!; trap 'kill $PF 2>/dev/null' EXIT; sleep 3
GW=http://127.0.0.1:18080
fails=0
check() { # name expected actual
  if [ "$2" = "$3" ]; then echo "PASS $1 ($3)"; else echo "FAIL $1: expected $2, got $3"; fails=$((fails+1)); fi
}
token() {
  curl -s -X POST "$GW/realms/eods/protocol/openid-connect/token" \
    -d grant_type=client_credentials -d client_id="$1" --data-urlencode client_secret="$(cat "$SECRETS/$2")" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin).get("access_token",""))'
}
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

READER=$(token eods-reader reader); FORECASTER=$(token eods-forecaster forecaster)
check "reader token issued" yes "$([ -n "$READER" ] && echo yes || echo no)"
check "forecaster token issued" yes "$([ -n "$FORECASTER" ] && echo yes || echo no)"
check "no token -> gateway 401" 401 "$(code $GW/v1/cube)"
check "garbage token -> 401" 401 "$(code -H 'Authorization: Bearer abc.def.ghi' $GW/v1/cube)"
check "reader reads cube" 200 "$(code -H "Authorization: Bearer $READER" $GW/v1/cube)"
check "reader cannot nowcast" 403 "$(code -X POST -H "Authorization: Bearer $READER" -H 'Content-Type: application/json' \
  -d '{"y1":128,"x1":128}' $GW/v1/nowcast)"
check "forecaster nowcasts" 200 "$(code -X POST -H "Authorization: Bearer $FORECASTER" -H 'Content-Type: application/json' \
  -d '{"y1":64,"x1":64}' $GW/v1/nowcast)"
check "window outside cube -> 422" 422 "$(code -X POST -H "Authorization: Bearer $FORECASTER" -H 'Content-Type: application/json' \
  -d '{"y1":100000,"x1":2}' $GW/v1/nowcast)"
check "admin console not routed" 404 "$(code $GW/admin/)"
check "metrics not routed" 404 "$(code $GW/metrics)"
check "wrong client secret refused" 401 "$(code -X POST $GW/realms/eods/protocol/openid-connect/token \
  -d grant_type=client_credentials -d client_id=eods-reader -d client_secret=wrong)"

# a pod in another namespace must not reach the API directly, bypassing the gateway
kubectl run np-probe --image=curlimages/curl:8.11.1 -n default --restart=Never --rm -i --quiet \
  --command -- curl -s -m 5 -o /dev/null -w '%{http_code}' http://eods-api.eods.svc.cluster.local:8000/healthz \
  > /tmp/np.out 2>/dev/null
check "direct pod-to-API blocked by NetworkPolicy" "000" "$(tr -d '\n' < /tmp/np.out | tail -c 3)"

echo; [ $fails -eq 0 ] && echo "all checks passed" || { echo "$fails check(s) failed"; exit 1; }
