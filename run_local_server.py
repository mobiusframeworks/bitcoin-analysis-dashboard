#!/usr/bin/env python3
"""
Local Web Server for Bitcoin Dashboard
Serves the dashboard on http://localhost:8000
"""

import http.server
import socketserver
import os
from pathlib import Path

# Configuration
PORT = 8000
DIRECTORY = Path(__file__).parent / "reports"

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def end_headers(self):
        # Disable caching for live updates
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def main():
    os.chdir(DIRECTORY)

    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("=" * 80)
        print("🚀 BITCOIN DASHBOARD - LOCAL SERVER")
        print("=" * 80)
        print(f"📊 Server running at: http://localhost:{PORT}")
        print(f"📁 Serving directory: {DIRECTORY}")
        print()
        print("💡 Dashboard will auto-refresh every 5 minutes")
        print("🔄 Data updates running in background (if started)")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 80)
        print()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped by user")
            print("=" * 80)

if __name__ == "__main__":
    main()
