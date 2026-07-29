import functools
import http.server
import socketserver

DIRECTORY = "/Users/gehnabhatia/threadlytics"
PORT = 4599

Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DIRECTORY)

with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
    httpd.serve_forever()
