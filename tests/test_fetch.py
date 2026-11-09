import io
import json

from zjxwx import fetch as f


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_get_retries_truncated_json(monkeypatch):
    body = json.dumps([{"hourly": {"cape": [1, 2, 3]}}]).encode()
    replies = [body[: len(body) // 2], body]
    monkeypatch.setattr(f.urllib.request, "urlopen", lambda url, timeout: _Resp(replies.pop(0)))
    monkeypatch.setattr(f.time, "sleep", lambda s: None)
    logs = []
    assert f._get({"a": 1}, "http://x", log=logs.append) == json.loads(body)
    assert any("retry" in m for m in logs)
