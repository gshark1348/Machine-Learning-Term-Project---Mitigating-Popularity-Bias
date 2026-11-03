#!/bin/zsh
set -e
cd "${0:A:h}"
if [[ ! -x .venv/bin/uvicorn ]]; then
  if command -v python3.11 >/dev/null; then python3.11 -m venv .venv; else python3 -m venv .venv; fi
  .venv/bin/python -m pip install -r requirements.txt
fi
echo '브라우저에서 http://127.0.0.1:18765 를 여세요. 종료하려면 Ctrl+C를 누르세요.'
exec .venv/bin/uvicorn app:api --host 127.0.0.1 --port 18765
