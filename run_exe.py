# Copyright (c) 2024 Microsoft Corporation.
import os
import socket
import sys
import time
import webbrowser
from subprocess import PIPE, STDOUT, Popen

PORT = 8503


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def main():
    # Getting path to python executable (full path of deployed python on Windows)
    executable = sys.executable
    os.environ["DB_APP_DATA"] = os.environ["LOCALAPPDATA"]
    os.environ["MODE"] = "exe"

    path_to_main = os.path.join(os.path.dirname(__file__), "app", "Home.py")

    port_use = is_port_in_use(PORT)
    print(f"Port {PORT} is in use: {port_use}")
    if port_use:
        webbrowser.open(f"http://localhost:{PORT}")
        return

    # Running streamlit server in a subprocess and writing to log file
    proc = Popen(
        [
            executable,
            "-m",
            "streamlit",
            "run",
            path_to_main,
            # The following option appears to be necessary to correctly start the streamlit server,
            # but it should start without it. More investigations should be carried out.
            "--server.headless=true",
            "--global.developmentMode=false",
            f"--server.port={PORT}",
        ],
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
    )

    proc.stdin.close()

    # Force the opening (does not open automatically) of the browser tab after a brief delay to let
    # the streamlit server start.
    time.sleep(10)
    webbrowser.open(f"http://localhost:{PORT}")

    while True:
        s = proc.stdout.read()
        if not s:
            break
        print(s, end="")

    proc.wait()


if __name__ == "__main__":
    main()
