import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler


last_prediction_data = []


def set_data(data):
    global last_prediction_data
    last_prediction_data = data


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            with open("visual/index.html", "r", encoding="utf-8") as f:
                html = f.read()
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/data":
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(last_prediction_data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def start_server():
    server = HTTPServer(("0.0.0.0", 8080), RequestHandler)
    print("Сервер запущен на http://localhost:8080")
    server.serve_forever()