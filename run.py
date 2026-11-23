#!/usr/bin/env python3
"""One-command launcher.

    python run.py

1. verifies the Python version and installs missing dependencies (into the
   active virtual environment, or into a private ``.venv`` created next to this
   file when none is active);
2. reuses ONNX weights already on this machine and downloads only what is
   missing, verifying every file against its SHA-256;
3. starts the API and dashboard on http://localhost:8000.

Only the standard library is imported before step 1 completes.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
VENV_DIR = ROOT / ".venv"
MIN_PYTHON = (3, 10)

# import name -> requirement it comes from (kept in step with requirements.txt)
REQUIRED_MODULES = {
    "numpy": "numpy",
    "cv2": "opencv-python-headless",
    "onnxruntime": "onnxruntime",
    "onnx": "onnx",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "cryptography": "cryptography",
    "multipart": "python-multipart",
    "qrcode": "qrcode",
    "websockets": "websockets",
}


def log(message: str) -> None:
    print(f"[fairface-guard] {message}", flush=True)


def fail(message: str, code: int = 1) -> "None":
    print(f"[fairface-guard] ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def missing_modules() -> list[str]:
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(os.environ.get("CONDA_PREFIX"))


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def pip_install(python: str) -> None:
    log(f"installing dependencies from {REQUIREMENTS.name} ...")
    result = subprocess.run(
        [python, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(REQUIREMENTS)]
    )
    if result.returncode != 0:
        fail(
            "dependency installation failed. Check your network connection, or install manually:\n"
            f"    {python} -m pip install -r {REQUIREMENTS}"
        )


def ensure_dependencies(argv: list[str], allow_install: bool) -> None:
    missing = missing_modules()
    if not missing:
        return
    packages = ", ".join(sorted(REQUIRED_MODULES[m] for m in missing))
    if not allow_install:
        fail(f"missing dependencies: {packages}. Re-run without --no-install or install them manually.")
    if in_virtualenv():
        log(f"missing dependencies: {packages}")
        pip_install(sys.executable)
        importlib.invalidate_caches()
        still = missing_modules()
        if still:
            fail(f"still missing after installation: {', '.join(still)}")
        return
    # No environment is active. Never install into the system interpreter:
    # create (or reuse) a private one and continue inside it.
    python = venv_python()
    if not python.exists():
        log(f"creating virtual environment in {VENV_DIR.name}/ ...")
        import venv

        try:
            venv.EnvBuilder(with_pip=True, clear=False).create(VENV_DIR)
        except Exception as exc:  # e.g. python3-venv missing on Debian/Ubuntu
            fail(f"could not create a virtual environment ({exc}). Install the 'venv' module for your Python.")
    if os.environ.get("FAIRFACE_BOOTSTRAPPED") == "1":
        fail(f"dependencies are still missing inside {VENV_DIR}: {packages}")
    check = subprocess.run(
        [str(python), "-c", "import importlib.util,sys;"
         f"sys.exit(any(importlib.util.find_spec(m) is None for m in {list(REQUIRED_MODULES)!r}))"]
    )
    if check.returncode != 0:
        pip_install(str(python))
    log(f"continuing inside {VENV_DIR.name}/")
    env = dict(os.environ, FAIRFACE_BOOTSTRAPPED="1")
    raise SystemExit(subprocess.call([str(python), str(Path(__file__).resolve()), *argv], env=env))


def provision_models(offline: bool) -> None:
    from backend import config
    from backend.engine import model_manager

    def progress(event: str, name: str, detail: str = "") -> None:
        log(f"model {name}: {event}{(' ' + detail) if detail else ''}")

    try:
        model_manager.ensure_models(config.models_dir(), progress=progress, offline=offline)
    except model_manager.ModelProvisioningError as exc:
        fail(str(exc))


def ensure_tls_certificate(data_dir: Path) -> tuple[Path, Path]:
    """Create (once) a self-signed certificate covering localhost and LAN IPs."""
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    from backend.main import lan_addresses

    tls_dir = data_dir / "tls"
    cert_path, key_path = tls_dir / "cert.pem", tls_dir / "key.pem"
    addresses = ["127.0.0.1", *lan_addresses()]
    if cert_path.exists() and key_path.exists():
        try:
            cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            covered = {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
            remaining = cert.not_valid_after_utc - datetime.datetime.now(datetime.timezone.utc)
            if set(addresses) <= covered and remaining > datetime.timedelta(days=7):
                return cert_path, key_path
        except Exception:
            pass  # unreadable or outdated: regenerate
    tls_dir.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "FairFace Guard (self-signed)")])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(a)) for a in addresses]
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    key_bytes = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    if key_path.exists():
        key_path.unlink()
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(key_bytes)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start FairFace Guard (API + dashboard).")
    parser.add_argument("--host", default=None,
                        help="bind address (default 127.0.0.1; FAIRFACE_HOST overrides)")
    parser.add_argument("--port", type=int, default=None, help="port (default 8000; FAIRFACE_PORT overrides)")
    parser.add_argument("--lan", action="store_true",
                        help="listen on all interfaces so other devices can reach the dashboard")
    parser.add_argument("--https", action="store_true",
                        help="serve over HTTPS with a self-signed certificate (needed for camera access "
                             "from other devices); implies --lan")
    parser.add_argument("--no-install", action="store_true", help="do not install missing dependencies")
    parser.add_argument("--offline", action="store_true", help="do not download models; fail if any is missing")
    parser.add_argument("--check", action="store_true",
                        help="verify dependencies and models, then exit without starting the server")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if sys.version_info < MIN_PYTHON:
        fail(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required (found {sys.version.split()[0]}).")
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    ensure_dependencies(argv, allow_install=not args.no_install)
    log("dependencies OK")
    provision_models(offline=args.offline)

    from backend import config

    settings = config.load_settings()
    lan = args.lan or args.https
    host = args.host or os.environ.get("FAIRFACE_HOST") or ("0.0.0.0" if lan else "127.0.0.1")
    port = args.port or settings.port
    os.environ["FAIRFACE_HOST"], os.environ["FAIRFACE_PORT"] = host, str(port)

    # Never expose the API to a network without authentication. If the
    # operator did not choose a key, mint one for this run and print it.
    generated_key = False
    # FAIRFACE_OPEN_ACCESS=1 is a deliberate opt-out for a short-lived demo
    # (anyone with the address can use the dashboard and see its data).
    open_access = os.environ.get("FAIRFACE_OPEN_ACCESS", "").strip().lower() in ("1", "true", "yes")
    if open_access and not settings.api_key:
        log("WARNING: open access: no API key, anyone who can reach this server can use it and see its data")
    elif host not in ("127.0.0.1", "localhost", "::1") and not settings.api_key:
        import secrets

        os.environ["FAIRFACE_API_KEY"] = secrets.token_urlsafe(18)
        generated_key = True

    providers = config.select_providers()
    log(f"ONNX Runtime providers: {', '.join(providers)}")
    if args.check:
        log("check complete")
        return

    scheme = "https" if args.https else "http"
    ssl_kwargs = {}
    if args.https:
        cert_path, key_path = ensure_tls_certificate(config.data_dir())
        ssl_kwargs = {"ssl_certfile": str(cert_path), "ssl_keyfile": str(key_path)}
        log("serving a self-signed certificate: browsers will ask you to accept it once per device")
    log(f"dashboard: {scheme}://localhost:{port}")
    if host not in ("127.0.0.1", "localhost", "::1"):
        from backend.main import lan_addresses

        for address in lan_addresses():
            log(f"on your network: {scheme}://{address}:{port}")
        if generated_key:
            log(f"API key for the dashboard: {os.environ['FAIRFACE_API_KEY']}")
        elif settings.api_key:
            log("API key: taken from FAIRFACE_API_KEY")
        if not args.https:
            log("NOTE: browsers on other devices only allow camera access over HTTPS; use --https "
                "or a TLS-terminating proxy for the audience call room")

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        log_level=os.environ.get("FAIRFACE_LOG_LEVEL", "info").lower(),
        # A full-resolution frame fits comfortably; anything larger is refused.
        ws_max_size=settings.max_upload_bytes + 1024,
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
