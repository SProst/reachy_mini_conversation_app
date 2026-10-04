"""Acquire Debian native development files inside the checkout without installing system packages."""

import os
import sys
import json
import platform
import subprocess
from pathlib import Path

from dev import DEV_CACHE


def main() -> None:
    """Download and extract native prerequisites for a Debian cloud workspace."""
    if sys.platform != "linux" or platform.freedesktop_os_release().get("ID") != "debian":
        raise SystemExit(
            "--local-native supports Debian; install native prerequisites for your OS as described in README"
        )
    codename = platform.freedesktop_os_release()["VERSION_CODENAME"]
    apt_root = DEV_CACHE / "apt"
    native_root = DEV_CACHE / "native"
    if (native_root / "usr").is_dir() and (apt_root / "versions.json").is_file():
        print(f"Reusing native prerequisites recorded in {apt_root / 'versions.json'}")
        return
    for directory in ["lists/partial", "archives/partial", "empty", "packages"]:
        (apt_root / directory).mkdir(parents=True, exist_ok=True)
    (apt_root / "sources.list").write_text(
        "\n".join(
            f"deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] {url} {suite} main"
            for url, suite in [
                ("https://deb.debian.org/debian", codename),
                ("https://deb.debian.org/debian", f"{codename}-updates"),
                ("https://security.debian.org/debian-security", f"{codename}-security"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (apt_root / "apt.conf").write_text(
        "\n".join(
            f'{key} "{value}";'
            for key, value in {
                "Dir::Etc::parts": apt_root / "empty",
                "Dir::Etc::main": "-",
                "Dir::Etc::sourcelist": apt_root / "sources.list",
                "Dir::Etc::sourceparts": apt_root / "empty",
                "Dir::State::lists": apt_root / "lists",
                "Dir::Cache::archives": apt_root / "archives",
                "Dir::Cache::pkgcache": apt_root / "pkgcache.bin",
                "Dir::Cache::srcpkgcache": apt_root / "srcpkgcache.bin",
                "Dir::Log": apt_root / "logs",
            }.items()
        )
        + "\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["APT_CONFIG"] = str(apt_root / "apt.conf")
    apt = "/usr/bin/apt-get"
    subprocess.run([apt, "update", "--error-on=any"], env=environment, check=True)
    required = ["libcairo2-dev", "libgirepository1.0-dev", "gir1.2-gst-plugins-base-1.0"]
    plan = subprocess.run(
        [apt, "--simulate", "--no-install-recommends", "install", *required],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (apt_root / "plan.txt").write_text(plan, encoding="utf-8")
    packages = sorted(set(required + [line.split()[1] for line in plan.splitlines() if line.startswith("Inst ")]))
    subprocess.run([apt, "download", *packages], cwd=apt_root / "packages", env=environment, check=True)
    versions = []
    for archive in sorted((apt_root / "packages").glob("*.deb")):
        subprocess.run(["dpkg-deb", "-x", str(archive), str(native_root)], check=True)
        versions.append(subprocess.check_output(["dpkg-deb", "-f", str(archive), "Package", "Version"], text=True))
    for link in native_root.rglob("*"):
        if link.is_symlink() and not link.exists():
            target = link.resolve()
            if target.is_relative_to(native_root):
                installed = Path("/") / target.relative_to(native_root)
                if installed.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(installed)
    (apt_root / "versions.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    print(f"Native prerequisites extracted to {native_root}; versions recorded in {apt_root / 'versions.json'}")


if __name__ == "__main__":
    main()
