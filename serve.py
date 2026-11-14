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
    ap.add_argument("--bridge-port", type=int, default=8778)
    ap.add_argument("--bridge", action="store_true",
                    help="also start the standalone blender bridge")
    a = ap.parse_args()
    if a.bridge:   # optional: only needed for non-Blender consumers
        import blender_bridge
        blender_bridge.start(a.bridge_port)
        print(f"blender bridge       -> http://localhost:{a.bridge_port}")
    print(f"isophote (client-side) -> http://localhost:{a.port}")
    http.server.ThreadingHTTPServer(
        ("127.0.0.1", a.port), Handler).serve_forever()
