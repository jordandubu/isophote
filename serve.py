#!/usr/bin/env python3
"""Static file server for the isophote web app (all computation client-side).
Only serves files; painting happens in the browser via pyodide."""
import http.server
import functools

Handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                            directory="app")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    a = ap.parse_args()
    print(f"isophote (client-side) -> http://localhost:{a.port}")
    http.server.ThreadingHTTPServer(
        ("127.0.0.1", a.port), Handler).serve_forever()
