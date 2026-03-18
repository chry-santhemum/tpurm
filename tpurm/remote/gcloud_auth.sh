# Shared GCS auth helper for remote scripts.

setup_gcloud_auth() {
  local log_prefix="${1:-[auth]}"

  if [[ -z "${SA_KEY_FILE:-}" && -n "${KEYS_DIR:-}" && -n "${REGION:-}" ]]; then
    SA_KEY_FILE="${KEYS_DIR%/}/bucket-${REGION}.json"
  fi
  if [[ -z "${SA_KEY_FILE:-}" ]]; then
    return 0
  fi
  if [[ ! -f "$SA_KEY_FILE" ]]; then
    echo "$log_prefix WARN: key file not found: $SA_KEY_FILE"
    return 0
  fi

  export SA_KEY_FILE
  export GOOGLE_APPLICATION_CREDENTIALS="$SA_KEY_FILE"

  if [[ -z "${SERVICE_ACCOUNT:-}" ]]; then
    return 0
  fi

  local current_account
  current_account="$(gcloud config get-value account 2>/dev/null || true)"
  if [[ "$current_account" != "$SERVICE_ACCOUNT" ]]; then
    echo "$log_prefix activating service account via $SA_KEY_FILE"
    gcloud auth activate-service-account --key-file="$SA_KEY_FILE"
  fi
}
