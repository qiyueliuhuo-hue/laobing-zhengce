import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from aiagent import ask
from cache import cache
from xiangliangjiansuo import clear_kb_cache

_BASE = os.path.dirname(os.path.abspath(__file__))

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(_BASE, "webapp.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("webapp")

MAX_MSG_LEN = 500  # 输入长度上限，防超长请求


def _ask(message: str, thread_id: str = "default", region: str = "", use_cache: bool = True) -> dict:
    t0 = time.time()
    try:
        r = ask(message, thread_id=thread_id, region=region, use_answer_cache=use_cache)
        log.info(f"chat ok uid={thread_id} region={region or '-'} "
                 f"cached={r.get('cached')} tools={len(r.get('trace', []))} "
                 f"len={len(r.get('reply', ''))} {time.time()-t0:.1f}s")
        return r
    except Exception as e:
        log.exception(f"chat error uid={thread_id}")
        return {"error": f"服务异常：{e}", "reply": "抱歉，回答时出错了，请稍后再试。"}


class Handler(BaseHTTPRequestHandler):
    server_version = "LaobingZhengce/1.0"

    def _send(self, code: int, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 1_000_000:
                return None, "body too large"
            raw = self.rfile.read(length)
            return json.loads(raw), None
        except Exception:
            return None, "bad json"

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            html_path = os.path.join(_BASE, "index.html")
            try:
                with open(html_path, "rb") as f:
                    html = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            except Exception as e:
                self._send(500, {"error": f"页面读取失败: {e}"})
        elif path == "/healthz":
            self._send(200, {"status": "ok", "ts": int(time.time())})
        elif path == "/api/stats":
            self._send(200, {"cache": cache.stats()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/chat":
            body, err = self._read_json()
            if err:
                return self._send(400, {"error": err})
            message = (body.get("message") or "").strip()
            user_id = (body.get("user_id") or "").strip() or "default"
            region = (body.get("region") or "").strip() or ""
            use_cache = bool(body.get("use_cache", True))
            if not message:
                return self._send(400, {"error": "empty message"})
            if len(message) > MAX_MSG_LEN:
                return self._send(400, {"error": f"message too long (> {MAX_MSG_LEN})"})
            return self._send(200, _ask(message, thread_id=user_id, region=region, use_cache=use_cache))
        elif path == "/api/clear_cache":
            # 政策库更新/重建后调用：清空两级缓存 + schema 缓存
            clear_kb_cache()
            return self._send(200, {"ok": True})
        elif path == "/api/clear_session":
            # 清除某用户记忆：前端"清除记忆"也走新 uid，这里提供显式删除
            body, err = self._read_json()
            if err:
                return self._send(400, {"error": err})
            uid = (body.get("user_id") or "").strip()
            if not uid:
                return self._send(400, {"error": "user_id required"})
            from aiagent import agent
            try:
                agent.delete_thread({"configurable": {"thread_id": uid}})
                log.info(f"session cleared uid={uid}")
                return self._send(200, {"ok": True})
            except Exception as e:
                return self._send(500, {"error": str(e)})
        self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):  # 自定义访问日志（含 IP 与状态）
        try:
            log.info("req %s %s -> %s" % (self.address_string(), fmt % args, self.log_request_code))
        except Exception:
            pass


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    log.info(f"老兵政策助手已启动: http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
