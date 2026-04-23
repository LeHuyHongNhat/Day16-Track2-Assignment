#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting lightweight CPU inference stub setup"

# AL2023 uses dnf. Install python runtime for a lightweight HTTP service.
dnf update -y
dnf install -y python3

cat >/opt/inference_stub.py <<'PY'
from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "mode": "cpu-stub"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._send_json(
                200,
                {
                    "id": "chatcmpl-cpu-stub",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "CPU stub is running. Replace this with your LightGBM inference service."
                            },
                            "finish_reason": "stop"
                        }
                    ]
                }
            )
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        return


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
PY

cat >/etc/systemd/system/inference-stub.service <<'EOF'
[Unit]
Description=Lightweight CPU inference stub
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /opt/inference_stub.py
Restart=always
RestartSec=3
User=root

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now inference-stub.service
echo "CPU inference stub started on port 8000"