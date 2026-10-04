"""Behavior checks for the isolated offline test entry point."""

import os
import sys
import socket
import textwrap
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _probe(tmp_path: Path, source: str, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    probe = tmp_path / "test_offline_probe.py"
    probe.write_text(textwrap.dedent(source), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "offline_pytest.py"), str(probe), "-q"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "operation",
    [
        "socket.getaddrinfo('offline-probe.invalid', 443)",
        "socket.gethostbyname('offline-probe.invalid')",
        "connection.connect(('192.0.2.1', 443))",
        "connection.sendto(b'probe', ('192.0.2.1', 443))",
    ],
)
def test_offline_runner_blocks_external_sockets(tmp_path: Path, operation: str) -> None:
    """External traffic fails at the audit hook before any network operation."""
    result = _probe(
        tmp_path,
        f"""
        import socket
        import pytest

        def test_external_request():
            with socket.socket() as connection:
                connection.settimeout(0.1)
                with pytest.raises(RuntimeError, match='External network disabled'):
                    {operation}
        """,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("socket_options_available", [True, False])
def test_offline_runner_allows_test_owned_loopback(tmp_path: Path, socket_options_available: bool) -> None:
    """A test can connect to the loopback server it created."""
    result = _probe(
        tmp_path,
        f"""
        import errno
        import socket

        def test_local_server(monkeypatch):
            def unsupported_socket_option(*args):
                raise OSError(errno.ENOPROTOOPT, 'Socket option unavailable')

            if not {socket_options_available}:
                monkeypatch.setattr(socket.socket, 'getsockopt', unsupported_socket_option)
            with socket.socket() as server, socket.socket() as client:
                server.bind(('127.0.0.1', 0))
                server.listen()
                client.settimeout(1)
                client.connect(server.getsockname())
                accepted, _ = server.accept()
                accepted.close()
        """,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("listener_state", ["bound", "closed"])
def test_offline_runner_rejects_inactive_listener(tmp_path: Path, listener_state: str) -> None:
    """Binding a socket or retaining a closed listener does not authorize connections."""
    result = _probe(
        tmp_path,
        f"""
        import socket
        import pytest

        def test_inactive_listener():
            with socket.socket() as server, socket.socket() as client:
                server.bind(('127.0.0.1', 0))
                address = server.getsockname()
                if {listener_state!r} == 'closed':
                    server.listen()
                    server.close()
                with pytest.raises(ConnectionRefusedError, match='No test-owned listener'):
                    client.connect(address)
        """,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_offline_runner_rejects_existing_loopback_service(tmp_path: Path) -> None:
    """A service owned by another process is not a permitted test listener."""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        result = _probe(
            tmp_path,
            f"""
            import socket
            import pytest

            def test_existing_service():
                with socket.socket() as client:
                    with pytest.raises(ConnectionRefusedError, match='No test-owned listener'):
                        client.connect(('127.0.0.1', {server.getsockname()[1]}))
            """,
        )
    assert result.returncode == 0, result.stdout + result.stderr


def test_offline_runner_isolates_configuration(tmp_path: Path) -> None:
    """Inherited credentials, endpoint overrides and app settings cannot affect a run."""
    environment = os.environ.copy()
    environment.update(
        {
            "OPENAI_API_KEY": "offline-probe-key",
            "HF_TOKEN": "offline-probe-token",
            "HF_REALTIME_WS_URL": "ws://offline-probe.invalid/v1",
            "REACHY_MINI_INSTANCE_PATH": str(tmp_path / "existing-instance"),
        }
    )
    result = _probe(
        tmp_path,
        """
        import os
        from pathlib import Path

        def test_configuration():
            for name in ['OPENAI_API_KEY', 'HF_TOKEN', 'HF_REALTIME_WS_URL']:
                assert name not in os.environ
            for name in ['UV_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'REACHY_MINI_SKIP_DOTENV']:
                assert os.environ[name] == '1'
            assert Path(os.environ['REACHY_MINI_INSTANCE_PATH']).name == 'instance'
            assert not Path(os.environ['HF_HOME']).exists()
        """,
        environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_validation_command_preserves_pytest_failure(tmp_path: Path) -> None:
    """A failing test makes the portable validation command fail too."""
    probe = tmp_path / "test_failing_probe.py"
    probe.write_text("def test_failure():\n    assert False\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "dev.py"), "test", str(probe), "-q"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed" in result.stdout
