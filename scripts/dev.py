"""Prepare dependencies online and run development checks offline."""

import os
import sys
import shlex
import shutil
import argparse
import sysconfig
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_CACHE = PROJECT_ROOT / "cache" / "dev"


def development_environment() -> dict[str, str]:
    """Return checkout-local dependency and native-library paths."""
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(DEV_CACHE / "uv")
    environment["UV_PROJECT_ENVIRONMENT"] = str(PROJECT_ROOT / ".venv")
    environment.pop("VIRTUAL_ENV", None)
    native_root = DEV_CACHE / "native"
    if sys.platform == "linux" and (native_root / "usr").is_dir():
        library_path = native_root / "usr" / "lib" / str(sysconfig.get_config_var("MULTIARCH"))
        environment["PKG_CONFIG_PATH"] = os.pathsep.join(
            [str(library_path / "pkgconfig"), str(native_root / "usr" / "share" / "pkgconfig")]
        )
        environment["PKG_CONFIG_SYSROOT_DIR"] = str(native_root)
        for variable, path in (
            ("LD_LIBRARY_PATH", library_path),
            ("GI_TYPELIB_PATH", library_path / "girepository-1.0"),
        ):
            environment[variable] = os.pathsep.join(filter(None, [str(path), environment.get(variable)]))
        environment["CC"] = "gcc"
        environment["CXX"] = "g++"
        environment["CFLAGS"] = "-I" + shlex.quote(str(native_root / "usr" / "include" / library_path.name))
    return environment


def main() -> int:
    """Acquire dependencies or run the selected checks without updating them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["setup", "check", "lint", "types", "lock", "test"])
    parser.add_argument("--local-native", action="store_true", help="Extract Debian native dependencies without sudo")
    parser.add_argument("--tools-only", action="store_true", help="Prepare only lint/type/test tools")
    arguments, pytest_arguments = parser.parse_known_args()
    if pytest_arguments and arguments.command != "test":
        parser.error("Extra arguments are accepted only by the test command")
    if (arguments.local_native or arguments.tools_only) and arguments.command != "setup":
        parser.error("Setup options require the setup command")
    if arguments.local_native and arguments.tools_only:
        parser.error("--local-native cannot be combined with --tools-only")
    uv = shutil.which("uv")
    if uv is None:
        parser.error("Install uv before running this script (see README)")
    if arguments.local_native:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "native_deps.py")], check=True)
    environment = development_environment()
    if arguments.command == "setup":
        environment.pop("UV_OFFLINE", None)
        command = [uv, "sync", "--frozen", "--python", "3.12"]
        command += ["--only-group", "dev", "--no-install-project"] if arguments.tools_only else ["--group", "dev"]
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
        if not arguments.tools_only:
            subprocess.run(
                [
                    "git",
                    "lfs",
                    "pull",
                    "--include=src/reachy_mini_conversation_app/static/avatars/*.svg",
                    "--exclude=",
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )
        return 0

    python = PROJECT_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        parser.error("Run python scripts/dev.py setup first")
    environment["UV_OFFLINE"] = "1"
    checks = ["lint", "types", "lock", "test"] if arguments.command == "check" else [arguments.command]
    commands = {
        "lint": [[str(python), "-m", "ruff", "check", "."], [str(python), "-m", "ruff", "format", "--check", "."]],
        "types": [[str(python), "-m", "mypy", "--pretty", "--show-error-codes"]],
        "lock": [[uv, "lock", "--check", "--offline"]],
        "test": [
            [str(python), str(PROJECT_ROOT / "scripts" / "offline_pytest.py"), *(pytest_arguments or ["tests/", "-v"])]
        ],
    }
    for check in checks:
        for command in commands[check]:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment)
            if completed.returncode:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
