#!/usr/bin/env bash
# Normalize and validate METHOD for upsampling submit (dry-run safe).
set -euo pipefail

_raw="${1:-}"
if [[ -z "${_raw}" ]]; then
  echo "Usage: $0 <pdans|punet|pugcn>" >&2
  exit 1
fi

# Strip whitespace and stray trailing brace from legacy bad exports.
METHOD="${_raw//[[:space:]]/}"
METHOD="${METHOD%\}}"

case "${METHOD}" in
  pdans|punet|pugcn)
    echo "${METHOD}"
    ;;
  *)
    echo "ERROR: invalid METHOD=${_raw!r} (normalized=${METHOD!r})" >&2
    exit 1
    ;;
esac
