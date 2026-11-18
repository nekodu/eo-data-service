#!/usr/bin/env bash
# Brings up the whole lab on kind: Envoy Gateway, Prometheus/Grafana, Keycloak, API, ingest.
set -euo pipefail
cd "$(dirname "$0")/.."

ENVOY_GATEWAY_VERSION=v1.5.1
KPS_VERSION=${KPS_VERSION:-}   # kube-prometheus-stack chart; empty = repo default
SECRETS=deploy/.secrets
mkdir -p "$SECRETS"; chmod 700 "$SECRETS"

gen() { [ -s "$SECRETS/$1" ] || head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32 > "$SECRETS/$1"; cat "$SECRETS/$1"; }
READER_SECRET=$(gen reader); FORECASTER_SECRET=$(gen forecaster)
KC_ADMIN=$(gen keycloak-admin); GRAFANA_ADMIN=$(gen grafana-admin)
export READER_SECRET FORECASTER_SECRET

export EODS_DATA_DIR=${EODS_DATA_DIR:-$HOME/eods-data}
mkdir -p "$EODS_DATA_DIR/models"
# hostPath ignores fsGroup, and the pods run as uid 10001, not the laptop user
chmod -R a+rwX "$EODS_DATA_DIR"
if ! kind get clusters | grep -qx eods; then
  envsubst '${EODS_DATA_DIR}' < deploy/kind.yaml > "$SECRETS/kind.yaml"
  kind create cluster --config "$SECRETS/kind.yaml" --wait 120s
fi
docker build -q -t eods:dev . >/dev/null
kind load docker-image eods:dev --name eods
# Keycloak is pulled by the node itself: kind load fails on its multi-arch manifest

helm upgrade --install eg oci://docker.io/envoyproxy/gateway-helm --version "$ENVOY_GATEWAY_VERSION" \
  -n envoy-gateway-system --create-namespace --wait --timeout 5m

# CI skips the monitoring stack to stay inside the runner's memory
if [ -z "${KPS_SKIP:-}" ]; then
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
  helm repo update prometheus-community >/dev/null
  kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
  kubectl -n monitoring create secret generic grafana-admin --from-literal=admin-user=admin \
    --from-literal=admin-password="$GRAFANA_ADMIN" --dry-run=client -o yaml | kubectl apply -f -
  helm upgrade --install kps prometheus-community/kube-prometheus-stack ${KPS_VERSION:+--version "$KPS_VERSION"} \
    -n monitoring -f deploy/monitoring/values.yaml --wait --timeout 10m
fi

kubectl apply -f deploy/k8s/namespace.yaml
kubectl -n eods create secret generic keycloak-admin --from-literal=username=admin \
  --from-literal=password="$KC_ADMIN" --dry-run=client -o yaml | kubectl apply -f -
envsubst '${READER_SECRET} ${FORECASTER_SECRET}' < deploy/keycloak/realm.json.tmpl > "$SECRETS/realm.json"
kubectl -n eods create secret generic keycloak-realm --from-file=eods-realm.json="$SECRETS/realm.json" \
  --dry-run=client -o yaml | kubectl apply -f -
if [ -f .env ]; then
  kubectl -n eods create secret generic eumetsat-credentials --from-env-file=.env \
    --dry-run=client -o yaml | kubectl apply -f -
else
  echo "no .env: ingest CronJob will fail until EUMETSAT credentials are added"
fi

kubectl apply -k deploy/k8s
if [ -z "${KPS_SKIP:-}" ]; then
  kubectl apply -f deploy/monitoring/eods-monitoring.yaml
  kubectl -n monitoring create configmap eods-dashboard --from-file=eods.json=deploy/monitoring/dashboard.json \
    --dry-run=client -o yaml | kubectl label --local -f - grafana_dashboard=1 -o yaml | kubectl apply -f -
fi

kubectl -n eods rollout status deploy/keycloak --timeout 5m
kubectl -n eods rollout status deploy/eods-api --timeout 3m || true
kubectl -n eods wait gateway/eods --for=condition=Programmed --timeout 3m
echo "up. run scripts/smoke.sh"
