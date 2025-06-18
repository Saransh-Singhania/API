# modular_server.py

from http.server import HTTPServer, BaseHTTPRequestHandler
import json

PORT = 8000

# --------- Handlers --------- #

def handle_root():
    return 200, {"message": "Hello, World!"}

def handle_ping():
    return 200, {"status": "ok"}

def handle_echo(body):
    try:
        data = json.loads(body)
        return 200, {"you_sent": data}
    except json.JSONDecodeError:
        return 400, {"error": "Invalid JSON"}

def handle_404():
    return 404, {"error": "Not Found"}

# --------- Request Router --------- #

def route_request(method, path, body=None):
    if method == "GET":
        if path == "/":
            return handle_root()
        elif path == "/ping":
            return handle_ping()
    elif method == "POST":
        if path == "/echo":
            return handle_echo(body)
    return handle_404()

# --------- HTTP Request Handler --------- #

def handle_request(request, client_address, server):
    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            status, response = route_request("GET", self.path)
            self.send_response(status)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def do_POST(self):
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode()
            status, response = route_request("POST", self.path, body)
            self.send_response(status)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    RequestHandler(request, client_address, server)

# --------- Start Server --------- #

def run_server():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, lambda req, addr, srv: handle_request(req, addr, srv))
    print(f"🚀 Serving on port {PORT}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
