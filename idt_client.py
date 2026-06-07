"""IDT SciTools Plus API client -- synthesis "complexity" scoring.

The complexity score is IDT's single industry-standard proxy for manufacturability /
synthesizability (higher = harder to synthesise). It is the x-axis of the Stage 2
quality-vs-manufacturability comparison.
    gene   : 7-19 = moderate (possible delay)
    gBlock : > 10 = rejected

Endpoints (validated against domesticator_3). Default host = ``sg.idtdna.com`` (the
instance an account is registered on; ``www`` returns 500 with the same credentials).
Each sequence returns a list of "issue" dicts; total score = sum of issue["Score"].
Rate limit: 500 requests / minute.

Security: credentials are never hard-coded. Pass a JSON file (domesticator format):
    {"ID": <client_id>, "secret": <client_secret>, "username": <login>,
     "password": <pw>, "token_file_path": <token cache>}

Example
-------
    from idt_client import IdtClient
    client = IdtClient("credentials.json", host="sg.idtdna.com", kind="gene")
    scores = client.score([("seq0", "ATG...")])   # -> {"seq0": (total, [(name, score, value), ...])}
"""

import json
import os
import sys
import threading
import time
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import error, parse, request

SCREEN_PATHS = {
    "gene":        "/Restapi/v1/Complexities/ScreenGeneSequences",       # full gene = our CDS
    "gblock":      "/Restapi/v1/Complexities/ScreenGblockSequences",
    "gblock_hifi": "/Restapi/v1/Complexities/ScreenGblockHifiSequences",
    "eblock":      "/Restapi/v1/Complexities/ScreenEblockSequences",
    "old":         "/api/complexities/screengBlockSequences",
}


def load_credentials(path):
    with open(os.path.expanduser(path)) as f:
        c = json.load(f)
    cr = {
        "client_id": c.get("ID") or c.get("client_id"),
        "client_secret": c.get("secret") or c.get("client_secret"),
        "username": c.get("username"),
        "password": c.get("password"),
        "token_file": os.path.expanduser(c.get("token_file_path") or c.get("token_file") or ""),
    }
    missing = [k for k in ("client_id", "client_secret", "username", "password") if not cr[k]]
    if missing:
        sys.exit(f"missing credentials: {missing} (required keys: ID, secret, username, password)")
    return cr


def total_complexity(entry):
    """One sequence result (list of issue dicts) -> (total, [(name, score, actual_value)]).
    Total = sum of issue scores."""
    if isinstance(entry, dict):
        for k in ("ComplexityScore", "Score", "score", "TotalScore"):
            if isinstance(entry.get(k), (int, float)):
                return float(entry[k]), [(str(entry.get("Name", k)), float(entry[k]), float("nan"))]
        entry = entry.get("Complexities") or entry.get("Issues") or entry.get("Results") or []
    total, issues = 0.0, []
    if isinstance(entry, list):
        for it in entry:
            if isinstance(it, dict):
                sc = it.get("Score", it.get("score"))
                sc = float(sc) if isinstance(sc, (int, float)) else 0.0
                total += sc
                name = str(it.get("Name") or it.get("DisplayText") or it.get("RuleName") or "?")
                av = it.get("ActualValue")
                issues.append((name, sc, float(av) if isinstance(av, (int, float)) else float("nan")))
    return total, issues


class _RateLimiter:
    """Thread-safe sliding-window limiter (<= ``max_per_min`` calls/minute)."""

    def __init__(self, max_per_min):
        self.max = max(int(max_per_min), 1)
        self.lock = threading.Lock()
        self.calls = []

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                self.calls = [t for t in self.calls if now - t < 60.0]
                if len(self.calls) < self.max:
                    self.calls.append(now)
                    return
                wait = 60.0 - (now - self.calls[0]) + 0.02
            time.sleep(max(wait, 0.01))


class IdtClient:
    """Authenticated, rate-limited, parallel complexity scorer."""

    def __init__(self, credentials, host="sg.idtdna.com", kind="gene", rate=450):
        self.cr = load_credentials(credentials) if isinstance(credentials, str) else credentials
        self.host = host
        self.token_url = f"https://{host}/Identityserver/connect/token"
        self.screen_url = f"https://{host}{SCREEN_PATHS[kind]}"
        self.kind = kind
        self.rl = _RateLimiter(rate)

    # -- auth --
    def _fetch_token(self):
        cr = self.cr
        auth = b64encode(f"{cr['client_id']}:{cr['client_secret']}".encode()).decode()
        data = parse.urlencode({"grant_type": "password", "scope": "test",
                                "username": cr["username"], "password": cr["password"]}).encode()
        req = request.Request(self.token_url, data=data, method="POST", headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "Basic " + auth})
        try:
            with request.urlopen(req, timeout=60) as r:
                tok = json.loads(r.read().decode())
        except error.HTTPError as e:
            sys.exit(f"token request failed HTTP {e.code}: {e.read().decode()[:400]}\n"
                     f"  (host={self.host}; check ID/secret/username, www<->sg instance mismatch -> 500)")
        tf = cr.get("token_file")
        if tf:
            try:
                d = os.path.dirname(tf)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(tf, "w") as f:
                    json.dump(tok, f)
                os.chmod(tf, 0o600)
            except Exception:
                pass
        return tok["access_token"]

    def _get_token(self):
        tf = self.cr.get("token_file")
        if tf and os.path.exists(tf):
            try:
                tok = json.load(open(tf))
                if time.time() - os.path.getmtime(tf) < tok.get("expires_in", 3600) - 60:
                    return tok["access_token"]
            except Exception:
                pass
        return self._fetch_token()

    # -- one request --
    def _screen(self, seqs, token):
        body = json.dumps([{"Name": n, "Sequence": s} for n, s in seqs]).encode()
        req = request.Request(self.screen_url, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + token})
        with request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode())

    @staticmethod
    def _parse(resp, chunk):
        results = resp
        if isinstance(resp, dict):
            for k in ("Result", "Results", "results", "data"):
                if isinstance(resp.get(k), list):
                    results = resp[k]
                    break
        if isinstance(results, list) and len(results) == len(chunk):
            return {sid: total_complexity(entry) for (sid, _), entry in zip(chunk, results)}, True
        return {sid: (float("nan"), []) for sid, _ in chunk}, False

    # -- public API --
    def score(self, named_seqs, batch=90, workers=8, debug_path=None):
        """``named_seqs`` = [(id, dna)] -> ``{id: (total, [issues])}``.

        Network-I/O bound, so threads (not processes) parallelise well. ``batch`` must be
        < 100 for the IDT ``gene`` endpoint.
        """
        batch = min(batch, 99)
        n = len(named_seqs)
        chunks = [named_seqs[i:i + batch] for i in range(0, n, batch)]
        tok = {"v": self._get_token()}
        tlock = threading.Lock()
        out, out_lock, done = {}, threading.Lock(), [0]

        def work(ci, chunk):
            payload = [(str(s), q) for s, q in chunk]
            resp = None
            for attempt in range(1, 8):
                self.rl.acquire()
                try:
                    resp = self._screen(payload, tok["v"])
                    break
                except error.HTTPError as e:
                    code, body = e.code, e.read().decode(errors="replace")[:120]
                    if code == 401 and attempt < 7:
                        with tlock:
                            tok["v"] = self._fetch_token()       # token expired -> refresh (shared)
                        continue
                    if code in (429, 500, 502, 503, 504) and attempt < 7:
                        time.sleep(15 if code != 429 else 10)
                        continue
                    print(f"    [warn] chunk{ci} HTTP {code}: {body}", flush=True)
                    break
                except Exception as ex:
                    if attempt < 7:
                        time.sleep(6)
                        continue
                    print(f"    [warn] chunk{ci} {ex}", flush=True)
                    break
            if debug_path and ci == 0 and resp is not None:
                try:
                    json.dump(resp, open(debug_path, "w"), indent=2, ensure_ascii=False)
                except Exception:
                    pass
            parsed, ok = self._parse(resp, chunk)
            if not ok and resp is not None:
                print(f"    [warn] chunk{ci} response count mismatch -> reduce --idt_batch", flush=True)
            return parsed

        with ThreadPoolExecutor(max_workers=max(workers, 1)) as ex:
            futs = {ex.submit(work, ci, c): len(c) for ci, c in enumerate(chunks)}
            for fut in as_completed(futs):
                with out_lock:
                    out.update(fut.result())
                    done[0] += futs[fut]
                    if done[0] % 500 < batch:
                        print(f"  idt scored {done[0]}/{n}", flush=True)
        return out
