"""
Agent Control Plane - Web Dashboard launcher.
Run: python -m control_plane.web
"""

from .server import run_server

if __name__ == "__main__":
    run_server(host="0.0.0.0", port=8090)
