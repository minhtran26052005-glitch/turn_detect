"""Use the existing VS Code Colab session through Jupyter's authenticated API.

Only google.colab's assigned-server secret is read. Credentials stay in memory;
they are never written into the project, notebook, URLs, or reports.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from urllib.parse import quote, urlparse
import uuid

import requests
import websocket


def _unprotect(value: bytes) -> bytes:
    class Blob(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]
    buffer = ctypes.create_string_buffer(value)
    source = Blob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    crypt32 = ctypes.windll.crypt32
    crypt32.CryptUnprotectData.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    if not crypt32.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 0,
                                      ctypes.byref(target)):
        raise OSError("Windows could not unlock the current user's VS Code Colab credential")
    try:
        return ctypes.string_at(target.data, target.length)
    finally:
        ctypes.windll.kernel32.LocalFree(ctypes.cast(target.data, ctypes.c_void_p))


def assigned_servers() -> list[dict]:
    if os.name != "nt":
        raise RuntimeError("VS Code credential discovery currently supports Windows only")
    code_root = Path(os.environ["APPDATA"]) / "Code"
    secret_key = 'secret://{"extensionId":"google.colab","key":"google.assigned_servers"}'
    database = code_root / "User" / "globalStorage" / "state.vscdb"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT value FROM ItemTable WHERE key=?", (secret_key,)).fetchone()
    if row is None:
        raise RuntimeError("No assigned Colab runtime. Connect a.ipynb using the Colab extension first.")
    encrypted = bytes(json.loads(row[0])["data"])
    if encrypted.startswith(b"v10"):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        state = json.loads((code_root / "Local State").read_text(encoding="utf-8"))
        protected_key = base64.b64decode(state["os_crypt"]["encrypted_key"])
        if not protected_key.startswith(b"DPAPI"):
            raise RuntimeError("Unsupported VS Code encryption format")
        key = _unprotect(protected_key[5:])
        clear = AESGCM(key).decrypt(encrypted[3:15], encrypted[15:], None)
    else:
        clear = _unprotect(encrypted)
    data = json.loads(clear.decode("utf-8"))
    if isinstance(data, list):
        return data
    for name in ("servers", "data", "assignments"):
        if isinstance(data.get(name), list):
            return data[name]
    raise RuntimeError("Unsupported assigned-server structure; no credentials were printed")


class ColabRuntime:
    def __init__(self, server_index: int = 0, session_path: str | None = None):
        servers = assigned_servers()
        if not servers:
            raise RuntimeError("The Colab extension has no assigned runtime")
        server = servers[server_index]
        self.label = server.get("label", "Colab")
        self.session_path = session_path
        information = server["connectionInformation"]
        raw_url = information["baseUrl"]
        if isinstance(raw_url, dict):
            raw_url = f"{raw_url['scheme']}://{raw_url['authority']}{raw_url.get('path', '/')}"
        self.base_url = raw_url.rstrip("/") + "/"
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or not parsed.hostname.endswith(".colab.dev"):
            raise RuntimeError("Refusing to send Colab credentials to an unexpected host")
        self.http = requests.Session()
        self.http.headers.update(information.get("headers") or {})
        self.http.headers["X-Colab-Runtime-Proxy-Token"] = information["token"]
        self.http.headers["X-Colab-Client-Agent"] = "vscode"
        self.kernel_id = None
        self.socket = None
        self.socket_session_id = uuid.uuid4().hex

    def request(self, method: str, path: str, **kwargs):
        response = self.http.request(method, self.base_url + path.lstrip("/"), timeout=120, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"Colab API {method} {path.split('?')[0]} returned HTTP {response.status_code}; "
                               "refresh the runtime connection in VS Code if it expired")
        return response

    def connect(self) -> dict:
        sessions = self.request("GET", "api/sessions").json()
        if self.session_path is not None:
            matches = [session for session in sessions if session.get("path") == self.session_path]
            if len(matches) != 1:
                raise RuntimeError("The explicitly selected notebook session is unavailable")
        else:
            matches = [session for session in sessions
                       if Path(session.get("path", "")).name == "a.ipynb"
                       or Path(session.get("path", "")).name.startswith("a-jvsc-")]
            if len(matches) != 1:
                if len(sessions) != 1:
                    raise RuntimeError("Select a.ipynb's runtime: multiple or no Jupyter sessions were found; "
                                       "use --session-path to select the notebook explicitly")
                matches = sessions
        kernel = matches[0]["kernel"]
        self.kernel_id = kernel["id"]
        return {"runtime_label": self.label, "notebook": matches[0].get("path"),
                "kernel_name": kernel.get("name"), "execution_state": kernel.get("execution_state")}

    def execute(self, code: str, timeout: int = 300, echo: bool = True) -> dict:
        if echo and os.name == "nt" and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if self.kernel_id is None:
            self.connect()
        message_id, session_id = uuid.uuid4().hex, self.socket_session_id
        socket_url = self.base_url.replace("https://", "wss://", 1)
        socket_url += f"api/kernels/{quote(self.kernel_id)}/channels?session_id={session_id}"
        headers = [f"{key}: {value}" for key, value in self.http.headers.items()
                   if key.lower() not in {"connection", "accept-encoding", "user-agent"}]
        if self.socket is None:
            # Retrying a handshake is safe: no execute_request has been sent.
            # Once code is sent, execution is never silently submitted again.
            for attempt in range(3):
                try:
                    self.socket = websocket.create_connection(socket_url, header=headers, timeout=30,
                                                               origin=self.base_url.rstrip("/"))
                    break
                except (websocket.WebSocketTimeoutException, OSError):
                    if attempt == 2:
                        raise
        socket = self.socket
        header = {"msg_id": message_id, "username": "turn_detect", "session": session_id,
                  "date": datetime.now(timezone.utc).isoformat(), "msg_type": "execute_request",
                  "version": "5.3"}
        request = {"header": header, "parent_header": {}, "metadata": {}, "channel": "shell",
                   "content": {"code": code, "silent": False, "store_history": False,
                               "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
                   "buffers": []}
        outputs, status, idle = [], None, False
        start = time.monotonic()
        try:
            socket.send(json.dumps(request))
            while time.monotonic() - start < timeout:
                socket.settimeout(min(10, max(0.1, timeout - (time.monotonic() - start))))
                try:
                    raw = socket.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if not raw:
                    raise ConnectionError("Colab WebSocket closed before execution completed")
                if isinstance(raw, bytes):
                    raise RuntimeError("Unexpected binary Jupyter protocol; no execution result was accepted")
                message = json.loads(raw)
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                kind, content = message["header"]["msg_type"], message["content"]
                if kind == "stream":
                    outputs.append({"output_type": "stream", **content})
                    if echo:
                        print(content["text"], end="", flush=True)
                elif kind in ("execute_result", "display_data"):
                    outputs.append({"output_type": kind, **content})
                    if echo and "text/plain" in content.get("data", {}):
                        print(content["data"]["text/plain"], flush=True)
                elif kind == "error":
                    outputs.append({"output_type": "error", **content})
                    if echo:
                        print(f"{content['ename']}: {content['evalue']}", flush=True)
                elif kind == "execute_reply":
                    status = content["status"]
                elif kind == "status" and content["execution_state"] == "idle":
                    idle = True
                if status is not None and idle:
                    return {"status": status, "outputs": outputs,
                            "elapsed_seconds": time.monotonic() - start}
            raise TimeoutError("Timed out waiting for this cell; the remote kernel was not interrupted")
        except BaseException:
            socket.close()
            self.socket = None
            raise

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def upload(self, local_path: Path, remote_path: str) -> None:
        parent = Path(remote_path).parent.as_posix()
        parts = parent.strip("/").split("/")
        for index in range(1, len(parts) + 1):
            path = "/".join(parts[:index])
            response = self.http.get(self.base_url + "api/contents/" + quote(path, safe="/"), timeout=30)
            if response.status_code == 404:
                self.request("PUT", "api/contents/" + quote(path, safe="/"), json={"type": "directory"})
            elif response.status_code >= 400:
                raise RuntimeError(f"Cannot inspect remote directory: HTTP {response.status_code}")
        self.request("PUT", "api/contents/" + quote(remote_path.strip("/"), safe="/"),
                     json={"type": "file", "format": "base64",
                           "content": base64.b64encode(local_path.read_bytes()).decode("ascii")})

    def download(self, remote_path: str, local_path: Path) -> None:
        value = self.request("GET", "api/contents/" + quote(remote_path.strip("/"), safe="/")
                             + "?format=base64").json()
        if value.get("type") != "file" or value.get("format") != "base64":
            raise ValueError("Expected a base64 file response")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(base64.b64decode(value["content"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-file", type=Path)
    parser.add_argument("--code")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--server-index", type=int, default=0)
    parser.add_argument("--session-path", help="Exact Jupyter notebook path when multiple sessions exist")
    args = parser.parse_args()
    runtime = ColabRuntime(args.server_index, args.session_path)
    connection = runtime.connect()
    print(json.dumps(connection, ensure_ascii=False), flush=True)
    code = args.code_file.read_text(encoding="utf-8-sig") if args.code_file else args.code
    code = code or "import json, os, platform, torch; print(json.dumps({'cwd':os.getcwd(), 'python':platform.python_version(), 'torch':torch.__version__, 'cuda':torch.cuda.is_available(), 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    result = runtime.execute(code, timeout=args.timeout)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps({"connection": connection, **result}, ensure_ascii=False,
                                         indent=2) + "\n", encoding="utf-8")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
