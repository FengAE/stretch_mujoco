"""Install the official Khronos KTX tools into the user's local cache."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.request
from pathlib import Path

import click


VERSION = "4.4.2"
URL = (
    "https://github.com/KhronosGroup/KTX-Software/releases/download/"
    "v4.4.2/KTX-Software-4.4.2-Linux-x86_64.deb"
)
SHA256 = "ca635ed489d8bf54fac8d7687056c651193de0740830a7738cc034adc63e3027"
DEFAULT_DESTINATION = Path.home() / ".cache" / "stretch_mujoco" / f"ktx-{VERSION}"


@click.command()
@click.option(
    "--destination",
    type=click.Path(path_type=Path),
    default=DEFAULT_DESTINATION,
    show_default=True,
)
def main(destination: Path) -> None:
    """Download, verify, and unpack Khronos KTX-Software for Linux x86-64."""
    executable = destination / "usr" / "bin" / "ktx"
    if executable.is_file():
        click.echo(f"KTX is already installed: {executable}")
        return
    if shutil.which("dpkg-deb") is None:
        raise click.ClickException("dpkg-deb is required to unpack the official KTX package")

    destination.parent.mkdir(parents=True, exist_ok=True)
    package = destination.parent / f"KTX-Software-{VERSION}-Linux-x86_64.deb"
    click.echo(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, package)
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    if digest != SHA256:
        package.unlink(missing_ok=True)
        raise click.ClickException(f"SHA-256 mismatch: expected {SHA256}, got {digest}")
    subprocess.run(["dpkg-deb", "-x", str(package), str(destination)], check=True)
    package.unlink(missing_ok=True)
    if not executable.is_file():
        raise click.ClickException("KTX package was extracted but the executable was not found")
    click.echo(f"Installed: {executable}")


if __name__ == "__main__":
    main()
