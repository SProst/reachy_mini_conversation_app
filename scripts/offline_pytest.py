"""Run pytest with isolated app settings and external Python sockets disabled."""

import os
import sys
import runpy
import socket
import logging
import tempfile
from pathlib import Path
from weakref import WeakSet

from dev import development_environment


logger = logging.getLogger(__name__)


def main() -> None:
    """Run tests without developer credentials, persistent app data, or model downloads."""
    environment = development_environment()
    for name in list(environment):
        if name.startswith(("REACHY_MINI_", "HF_", "HUGGING_FACE_", "OPENAI_")) or name == "AUTOLOAD_EXTERNAL_TOOLS":
            del environment[name]
    with tempfile.TemporaryDirectory(prefix="reachy-offline-") as directory:
        environment.update(
            {
                "UV_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HOME": str(Path(directory) / "hf"),
                "REACHY_MINI_SKIP_DOTENV": "1",
                "REACHY_MINI_INSTANCE_PATH": str(Path(directory) / "instance"),
            }
        )
        os.environ.clear()
        os.environ.update(environment)
        bound_sockets: WeakSet[socket.socket] = WeakSet()

        def guard_network(event: str, arguments: tuple[object, ...]) -> None:
            if event == "socket.bind" and isinstance(arguments[0], socket.socket):
                bound_sockets.add(arguments[0])
            if event == "socket.connect":
                address = arguments[1]
                if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
                    for listener in list(bound_sockets):
                        try:
                            if listener.getsockname()[:2] == address[:2] and listener.getsockopt(
                                socket.SOL_SOCKET, socket.SO_ACCEPTCONN
                            ):
                                return
                        except OSError as error:
                            logger.debug("Test listener is no longer available: %s", error)
                    raise ConnectionRefusedError("No test-owned listener at this loopback address")
            if event == "socket.getaddrinfo" and arguments[0] in {"127.0.0.1", "::1"}:
                return
            if event in {
                "socket.connect",
                "socket.sendto",
                "socket.getaddrinfo",
                "socket.gethostbyname",
                "socket.gethostbyaddr",
            }:
                raise RuntimeError(f"External network disabled during offline validation: {event}")

        sys.addaudithook(guard_network)
        sys.argv = ["pytest", *(sys.argv[1:] or ["tests/", "-v"])]
        runpy.run_module("pytest", run_name="__main__")


if __name__ == "__main__":
    main()
