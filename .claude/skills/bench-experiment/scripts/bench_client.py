#!/usr/bin/env python3
"""Tiny client for the Lab Bench API — submit jobs and poll, so heavy ops don't block the caller
and concurrent calls don't contend on the single GPU (which returns 500s).

Works against a local dev server (default http://127.0.0.1:7870) or a Modal `web_gui` URL
(https://<workspace>--qwen-scope-steering-gui-web-gui-dev.modal.run). Set BENCH_URL to choose.
SSL verification is disabled so it works against Modal dev URLs even when the local Python lacks a
CA bundle — fine for these self-hosted/own-deployment endpoints.

Examples:
  BENCH_URL=http://127.0.0.1:7870 python bench_client.py status
  python bench_client.py job benchmark '{"prompt_set":"{\\"id\\":\\"p1\\",\\"prompt\\":\\"Hi\\"}","feature_id":42,"strength":8,"layer":12,"max_new_tokens":12}'
  python bench_client.py job monitor_discover '{"behavior":"refusal","positive_examples":"No.\\nI can't.","negative_examples":"Sure.\\nYes.","layer":12,"top_k":3}'
  python bench_client.py get /api/experiments
  python bench_client.py post /api/monitor/score '{"text":"I cannot help.","features":[123],"layer":12,"threshold":0.5}'
"""
import json
import os
import ssl
import sys
import time
import urllib.request as U

BASE = os.environ.get("BENCH_URL", "http://127.0.0.1:7870").rstrip("/")
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def _call(method, path, body=None, timeout=300, tries=4):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"content-type": "application/json"} if data is not None else {}
    last = None
    for i in range(tries):
        try:
            req = U.Request(BASE + path, data=data, method=method, headers=headers)
            return json.loads(U.urlopen(req, timeout=timeout, context=_CTX).read())
        except Exception as exc:  # transient 408/5xx or cold-load timeouts — back off and retry
            last = exc
            time.sleep(2 * (i + 1))
    raise last


def get(path):
    return _call("GET", path)


def post(path, body):
    return _call("POST", path, body)


def run_job(op, params, poll_every=1.0, timeout=900):
    """Submit a job and poll until done/error — the correct way to run heavy model ops."""
    jid = post("/api/jobs", {"op": op, "params": params})["job_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = get(f"/api/jobs/{jid}")
        if job["status"] in ("done", "error"):
            return job
        time.sleep(poll_every)
    return get(f"/api/jobs/{jid}")


def main(argv):
    if not argv or argv[0] == "status":
        print(json.dumps(get("/api/status"), indent=2))
        return 0
    cmd = argv[0]
    if cmd == "job":
        op = argv[1]
        params = json.loads(argv[2]) if len(argv) > 2 else {}
        print(json.dumps(run_job(op, params), indent=2))
    elif cmd == "get":
        print(json.dumps(get(argv[1]), indent=2))
    elif cmd == "post":
        print(json.dumps(post(argv[1], json.loads(argv[2]) if len(argv) > 2 else {}), indent=2))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
