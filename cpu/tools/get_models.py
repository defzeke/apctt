"""Download catalogue models into a local store with the organizer's layout (decision D64).

    python tools/get_models.py --list [--json]
    python tools/get_models.py ID [ID ...] --store DIR [--read-only] [--hf-endpoint URL] [--json]
    python tools/get_models.py --all --store DIR [--read-only] [--hf-endpoint URL] [--json]
    python tools/get_models.py --check --store DIR [ID ... | --all] [--deep] [--json]
    python tools/get_models.py --snippet ID

The organizer provides the public models of contracts/model-catalog.json on every evaluation
machine in a read-only store, <APCTT_MODEL_STORE>/<org>/<name>/<revision>/, and gives the worker
its path in the environment variable APCTT_MODEL_STORE. This tool builds the same store on your
machine for development: for each model, only the files the catalogue lists (plus the licence and
notice files at the top of the repository), at the pinned revision, from the Hugging Face API
(https://huggingface.co; no token, the catalogue has no gated models). Every file is verified (its
size, the LFS sha256 of weight files, which must also be the catalogue's "weights" record, the git
blob id of the others) before the folder is moved into place, so a failed download leaves nothing
half-done. A repository without a licence file gets the
standard Apache-2.0 or MIT text from the kit's licenses/ folder as LICENSE. .apctt-model.json
records the files and their sha256. A complete copy is reused. --read-only removes the write bits,
as on the organizer's machines. --check reads an existing store without the network (--deep also
re-hashes every file). After a download the tool prints the APCTT_MODEL_STORE value for
config/my.local.json ("worker": {"extra_env": {"APCTT_MODEL_STORE": ...}}) and a loading snippet:
load from the local path with local_files_only and set HF_HUB_OFFLINE in your code, because the
evaluation runs without network (D46).

Exit codes: 0 done, 1 a download or check failed, 2 invalid catalogue or unknown model id, 3 usage
or tool error. Standard library only (plain HTTPS; the certifi bundle is used when installed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import shutil
import ssl
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import model_catalog  # noqa: E402 - the kit's catalogue loader

HF_ENDPOINT = "https://huggingface.co"
LICENCE_TEXTS_FILES = ("LICENSE*", "LICENCE*", "COPYING*", "License*", "license*")
LICENCE_FILES = (*LICENCE_TEXTS_FILES, "NOTICE*")  # fetched with the weights
LICENCE_TEXTS = {"Apache-2.0": "licenses/Apache-2.0.txt", "MIT": "licenses/MIT.txt"}
TIMEOUT_S = 60
USER_AGENT = "apctt-get-models"
EXIT_OK, EXIT_FAILED, EXIT_INVALID, EXIT_USAGE = 0, 1, 2, 3


class ModelError(Exception):
    """A model that cannot be fetched or verified."""


class UsageError(Exception):
    """A bad command line (exit code 3)."""


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1000
    return f"{size:.1f} GB"


# ------------------------------------------------------------------------------ HTTP


def ssl_context() -> ssl.SSLContext:
    """The system's certificates, or certifi's bundle when it is installed (python.org builds
    of Python on macOS have no certificates until certifi is used)."""
    try:
        import certifi  # noqa: PLC0415 - optional

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def request(url: str) -> urllib.request.Request:
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise ModelError(f"{url}: only http and https endpoints")
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310


def open_url(url: str) -> Any:
    context = ssl_context() if url.startswith("https:") else None
    return urllib.request.urlopen(request(url), timeout=TIMEOUT_S, context=context)  # noqa: S310


def fetch_json(url: str) -> Any:
    try:
        with open_url(url) as response:
            return json.loads(response.read())
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise ModelError(f"{url}: {exc}") from None


def safe_path(name: object) -> str:
    if not isinstance(name, str) or not name:
        raise ModelError("the repository lists a file without a name")
    if model_catalog.unsafe_path(name):
        raise ModelError(f"the repository lists an unsafe file path {name!r}")
    return name


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None  # LFS files
    blob: str | None  # git blob id of the other files


def remote_files(info: Any) -> list[RemoteFile]:
    files = []
    for sibling in info.get("siblings", []) if isinstance(info, dict) else []:
        if not isinstance(sibling, dict):
            continue
        path = safe_path(sibling.get("rfilename"))
        lfs = sibling.get("lfs") if isinstance(sibling.get("lfs"), dict) else None
        size = (lfs or sibling).get("size")
        if not isinstance(size, int) or size < 0:
            raise ModelError(f"{path}: the repository gives no size")
        files.append(
            RemoteFile(
                path,
                size,
                str(lfs["sha256"]) if lfs and lfs.get("sha256") else None,
                None if lfs else str(sibling.get("blobId") or "") or None,
            )
        )
    return files


def is_licence(path: str) -> bool:
    """A licence or notice file at the top of the repository (fetched with the weights)."""
    return "/" not in path and any(fnmatch.fnmatchcase(path, p) for p in LICENCE_FILES)


def is_licence_text(path: str) -> bool:
    """A licence text (a NOTICE file alone is not one)."""
    return "/" not in path and any(fnmatch.fnmatchcase(path, p) for p in LICENCE_TEXTS_FILES)


def select(files: Sequence[RemoteFile], wanted: Sequence[str]) -> list[RemoteFile]:
    """The catalogue's files, and the licence and notice files at the top of the repository."""
    by_path = {f.path: f for f in files}
    missing = [name for name in wanted if name not in by_path]
    if missing:
        raise ModelError(f"not in the repository at this revision: {', '.join(missing)}")
    chosen = {name: by_path[name] for name in wanted}
    chosen.update({f.path: f for f in files if is_licence(f.path)})
    return [chosen[name] for name in sorted(chosen)]


def download(url: str, target: Path, remote: RemoteFile) -> str:
    """Stream one file and verify it; returns its sha256."""
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(b"blob %d\0" % remote.size, usedforsecurity=False)
    size = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open_url(url) as response, target.open("wb") as sink:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                sha256.update(chunk)
                blob.update(chunk)
                size += len(chunk)
                sink.write(chunk)
    except (OSError, urllib.error.URLError) as exc:
        raise ModelError(f"{remote.path}: download failed ({exc})") from None
    if size != remote.size:
        raise ModelError(f"{remote.path}: {size} bytes, the repository says {remote.size}")
    if remote.sha256 is not None and sha256.hexdigest() != remote.sha256:
        raise ModelError(f"{remote.path}: sha256 differs from the repository's LFS record")
    if remote.sha256 is None and remote.blob and blob.hexdigest() != remote.blob:
        raise ModelError(f"{remote.path}: content differs from the repository's git blob id")
    return sha256.hexdigest()


# ------------------------------------------------------------------------------- store


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_read_only(path: Path) -> None:
    for directory, _, files in os.walk(path, topdown=False):
        for name in files:
            target = Path(directory) / name
            if not target.is_symlink():
                mode = target.stat().st_mode
                os.chmod(target, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        os.chmod(directory, 0o555)  # noqa: S103 - read-only, like the organizer's store


def remove_tree(path: Path) -> None:
    """Remove a model folder this tool made, read-only or not."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    for directory, _, files in os.walk(path):
        os.chmod(directory, 0o755)  # noqa: S103 - the owner removes the tree
        for name in files:
            target = Path(directory) / name
            if not target.is_symlink():
                os.chmod(target, target.stat().st_mode | stat.S_IWUSR)
    shutil.rmtree(path)


def read_record(folder: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((folder / model_catalog.MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def check_model(store: Path, model: model_catalog.Model, deep: bool = False) -> list[str]:
    """Problems of the store's copy of a catalogue model (empty: complete and as recorded)."""
    folder = model.directory(store)
    if not folder.is_dir():
        return [f"{model.relative().as_posix()} is not in the store"]
    record = read_record(folder)
    if record is None:
        return [f"{model.relative().as_posix()} has no readable {model_catalog.MARKER}"]
    problems = []
    for key, value in (("repo", model.repo), ("revision", model.revision), ("id", model.id)):
        if record.get(key) != value:
            problems.append(f"{model_catalog.MARKER} says {key} {record.get(key)!r}, not {value}")
    files: dict[str, dict[str, Any]] = {
        str(f.get("path")): f for f in record.get("files", []) if isinstance(f, dict)
    }
    for name in model.files:
        if name not in files:
            problems.append(f"{name} is not in the store record")
    for name, (size, sha256) in sorted(model.weights.items()):
        entry = files.get(name, {})
        if name in files and (entry.get("size"), entry.get("sha256")) != (size, sha256):
            problems.append(
                f"{name}: the store record differs from the catalogue's size and sha256"
            )
    for name, entry in sorted(files.items()):
        path = folder / name
        if not path.is_file():
            problems.append(f"{name} is missing")
        elif path.stat().st_size != entry.get("size"):
            problems.append(f"{name} is {path.stat().st_size} bytes, recorded {entry.get('size')}")
        elif deep and file_sha256(path) != entry.get("sha256"):
            problems.append(f"{name}: sha256 differs from the store record")
    if not any(is_licence_text(name) and (folder / name).is_file() for name in files):
        problems.append("no LICENSE file next to the weights")
    return problems


def licence_file(model: model_catalog.Model) -> bytes:
    """The licence text added to a copy whose repository has no licence file."""
    source = LICENCE_TEXTS.get(model.license)
    if source is None:
        raise ModelError(f"no licence text in the kit for {model.license}")
    text = (ROOT / source).read_text(encoding="utf-8")
    header = (
        f"Licence text added to this copy of {model.repo} at revision {model.revision}.\n"
        f"The repository declares {model.license} in its model card metadata but has no licence\n"
        f"file; the standard {model.license} text follows ({source} of the APCTT kit). The\n"
        "copyright holders are the authors named in the model card.\n\n"
    )
    return (header + text).encode("utf-8")


def fetch_model(
    store: Path,
    model: model_catalog.Model,
    endpoint: str = HF_ENDPOINT,
    read_only: bool = False,
    catalog: model_catalog.Catalog | None = None,
) -> dict[str, Any]:
    """Reuse a complete copy, else download it into <store>/<org>/<name>/<revision>/."""
    target = model.directory(store)
    result: dict[str, Any] = {"id": model.id, "repo": model.repo, "revision": model.revision,
                              "path": str(target)}  # fmt: skip
    if not check_model(store, model):
        record = read_record(target) or {}
        size = sum(int(f.get("size", 0)) for f in record.get("files", []))
        return {**result, "status": "reused", "bytes": size, "problems": []}
    quoted = urllib.parse.quote(model.repo, safe="/")
    info = fetch_json(f"{endpoint}/api/models/{quoted}/revision/{model.revision}?blobs=true")
    if not isinstance(info, dict) or info.get("sha") != model.revision:
        found = info.get("sha") if isinstance(info, dict) else None
        raise ModelError(f"the endpoint answered revision {found}, not {model.revision}")
    files = select(remote_files(info), model.files)
    for remote in files:
        expected = model.weights.get(remote.path)
        if expected is not None and (remote.size, remote.sha256) != expected:
            raise ModelError(
                f"{remote.path}: the repository's size and LFS sha256 differ from the catalogue's "
                "weights record"
            )
    partial = target.with_name(target.name + ".partial")
    remove_tree(partial)
    try:
        fetched: list[dict[str, Any]] = []
        for remote in files:
            url = f"{endpoint}/{quoted}/resolve/{model.revision}/{urllib.parse.quote(remote.path)}"
            digest = download(url, partial / remote.path, remote)
            fetched.append({"path": remote.path, "size": remote.size, "sha256": digest,
                            "origin": "repository"})  # fmt: skip
        licences = [f["path"] for f in fetched if is_licence(str(f["path"]))]
        added = None
        if not any(is_licence_text(str(name)) for name in licences):
            data = licence_file(model)
            (partial / "LICENSE").write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            fetched.append({"path": "LICENSE", "size": len(data), "sha256": digest,
                            "origin": "kit"})  # fmt: skip
            added = {"path": "LICENSE", "source": LICENCE_TEXTS[model.license], "sha256": digest}
        record = {
            "record_version": model_catalog.STORE_RECORD_VERSION,
            "id": model.id,
            "repo": model.repo,
            "revision": model.revision,
            "license": model.license,
            "files": sorted(fetched, key=lambda f: str(f["path"])),
            "licence_files": licences,
            "licence_added": added,
            "catalog_version": catalog.version if catalog else model_catalog.CATALOG_VERSION,
            "endpoint": endpoint,
            "fetched_at": utc_now(),
            "read_only": read_only,
        }
        (partial / model_catalog.MARKER).write_text(json.dumps(record, indent=2) + "\n", "utf-8")
        if read_only:
            make_read_only(partial)
        remove_tree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial.rename(target)
    except BaseException:
        remove_tree(partial)
        raise
    size = sum(int(f["size"]) for f in fetched)
    return {**result, "status": "downloaded", "bytes": size, "problems": []}


def get(
    store: Path,
    models: Sequence[model_catalog.Model],
    endpoint: str,
    read_only: bool,
    catalog: model_catalog.Catalog,
) -> list[dict[str, Any]]:
    results = []
    for model in models:
        try:
            results.append(fetch_model(store, model, endpoint, read_only, catalog))
        except ModelError as exc:
            results.append({"id": model.id, "repo": model.repo, "revision": model.revision,
                            "path": str(model.directory(store)), "status": "failed",
                            "problems": [str(exc)]})  # fmt: skip
    return results


def check_store(
    store: Path, models: Sequence[model_catalog.Model], deep: bool, explicit: bool
) -> list[dict[str, Any]]:
    results = []
    for model in models:
        folder = model.directory(store)
        if not folder.exists() and not explicit:
            results.append({"id": model.id, "path": str(folder), "status": "absent",
                            "problems": []})  # fmt: skip
            continue
        problems = check_model(store, model, deep)
        status = "failed" if problems else "ok"
        results.append({"id": model.id, "path": str(folder), "status": status,
                        "problems": problems})  # fmt: skip
    return results


# ------------------------------------------------------------------------------------ CLI


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def parser() -> Parser:
    p = Parser(
        prog="get_models.py",
        description=(__doc__ or "").split("\n\n")[0],
        epilog="Exit codes: 0 done, 1 download or check failed, 2 invalid catalogue or id, "
        "3 usage or tool error.",
    )
    p.add_argument("ids", nargs="*", metavar="ID", help="catalogue ids (see --list)")
    p.add_argument("--list", action="store_true", help="list the catalogue")
    p.add_argument("--all", action="store_true", help="every catalogue model")
    p.add_argument("--check", action="store_true", help="check an existing store (no network)")
    p.add_argument("--deep", action="store_true", help="with --check: re-hash every file")
    p.add_argument("--snippet", metavar="ID", help="print the loading snippet of a model")
    p.add_argument("--store", type=Path, help="the model store folder (APCTT_MODEL_STORE)")
    p.add_argument("--read-only", action="store_true", help="remove the write bits after download")
    p.add_argument("--hf-endpoint", default=HF_ENDPOINT, help="Hugging Face endpoint")
    p.add_argument("--catalog", type=Path, default=model_catalog.CATALOG, help=argparse.SUPPRESS)
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    return p


def chosen(args: argparse.Namespace, catalog: model_catalog.Catalog) -> list[model_catalog.Model]:
    if args.all:
        return list(catalog.models)
    unknown = [i for i in args.ids if catalog.get(i) is None]
    if unknown:
        raise model_catalog.CatalogError(
            f"not in the model catalogue: {', '.join(unknown)} (see python tools/get_models.py "
            "--list); other public models are not available (D64)"
        )
    return [m for m in catalog.models if m.id in args.ids]


def listing(catalog: model_catalog.Catalog) -> dict[str, Any]:
    rows = []
    for model in catalog.models:
        raw = model.raw
        rows.append({"id": model.id, "repo": model.repo, "revision": model.revision,
                     "license": model.license, "category": model.category,
                     "params_b": raw["params_b"], "download_bytes": model.download_bytes,
                     "vram_peak_gib": model.vram["peak"], "cpu_feasible": raw["cpu_feasible"],
                     "quantization": (model.load["quantization"] or {}).get("method"),
                     "measured": raw.get("measured"), "use": raw["use"]})  # fmt: skip
    return {**catalog.identity(), "gpu_budget": catalog.gpu_budget, "models": rows}


def store_lines(store: Path) -> list[str]:
    extra = json.dumps({"extra_env": {model_catalog.STORE_ENV: str(store)}})
    return [
        f"{model_catalog.STORE_ENV}={store}",
        f'config/my.local.json "worker": {extra}',
        f"export {model_catalog.STORE_ENV}={store}",
    ]


def execute(args: argparse.Namespace) -> tuple[int, dict[str, Any], list[str]]:
    """(exit code, JSON result, text lines)."""
    modes = sum(bool(x) for x in (args.list, args.check, args.snippet))
    if modes > 1:
        raise UsageError("give one of --list, --check, --snippet or model ids")
    if args.all and args.ids:
        raise UsageError("give model ids or --all, not both")
    catalog = model_catalog.load(args.catalog)
    if args.list:
        data = listing(catalog)
        lines = [f"{'id':28} {'download':>9} {'VRAM peak':>10}  cpu   use"]
        for row in data["models"]:
            cpu = str(row["cpu_feasible"]).split(":", 1)[0]
            kind = "[nf4] " if row["quantization"] == "bitsandbytes-nf4" else ""
            size, peak = human(row["download_bytes"]), row["vram_peak_gib"]
            lines.append(f"{row['id']:28} {size:>9} {peak:>6.2f} GiB  {cpu:5} {kind}{row['use']}")
        budget = catalog.gpu_budget
        lines.append(f"VRAM peak: estimates for the reference GPU {budget['reference_gpu']} "
                     f"(target {budget['target_peak_gib']} GiB) until the organizer measures them; "
                     "nf4 = 4-bit bitsandbytes, GPU only")  # fmt: skip
        lines.append(f"catalogue {catalog.version} ({catalog.status}), {len(data['models'])} "
                     "models; download: python tools/get_models.py ID --store DIR")  # fmt: skip
        return EXIT_OK, data, lines
    if args.snippet:
        model = catalog.get(args.snippet)
        if model is None:
            raise model_catalog.CatalogError(f"not in the model catalogue: {args.snippet}")
        snippet = model_catalog.loading_snippet(model)
        return EXIT_OK, {"id": model.id, "snippet": snippet}, snippet.rstrip("\n").splitlines()
    if args.store is None:
        raise UsageError("give --store DIR")
    store = args.store.absolute()
    if args.check:
        explicit = bool(args.ids or args.all)
        models = chosen(args, catalog) if explicit else list(catalog.models)
        results = check_store(store, models, args.deep, explicit)
        ok = all(r["status"] != "failed" for r in results)
        lines = [f"{r['status'].upper():8} {r['id']} {'; '.join(r['problems'])}".rstrip()
                 for r in results]  # fmt: skip
        lines.append(f"store {store}: {'OK' if ok else 'FAILED'}")
        data = {"store": str(store), "deep": args.deep, "models": results, "ok": ok}
        return (EXIT_OK if ok else EXIT_FAILED), data, lines
    if not args.ids and not args.all:
        raise UsageError("give model ids or --all (see --list)")
    models = chosen(args, catalog)
    endpoint = str(args.hf_endpoint).rstrip("/")
    results = get(store, models, endpoint, args.read_only, catalog)
    ok = all(r["status"] != "failed" for r in results)
    lines = []
    for result in results:
        size = human(result["bytes"]) if "bytes" in result else ""
        problems = "; ".join(result["problems"])
        lines.append(f"{result['status'].upper():10} {result['id']} -> {result['path']} {size} "
                     f"{problems}".rstrip())  # fmt: skip
    if ok:
        lines += store_lines(store)
        if not args.all:
            for model in models:
                lines += ["", f"# loading snippet for {model.id}"]
                lines += model_catalog.loading_snippet(model).rstrip("\n").splitlines()
        else:
            lines.append("loading snippet of a model: python tools/get_models.py --snippet ID")
    data = {"store": str(store), "env": {model_catalog.STORE_ENV: str(store)},
            "models": results, "ok": ok}  # fmt: skip
    return (EXIT_OK if ok else EXIT_FAILED), data, lines


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not (args.list or args.check or args.snippet or args.ids or args.all):
        parser().print_usage(sys.stderr)
        print("get_models.py: nothing to do (try --list)", file=sys.stderr)
        return EXIT_USAGE
    try:
        code, data, lines = execute(args)
    except UsageError as exc:
        print(f"get_models.py: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except model_catalog.CatalogError as exc:
        if args.json:
            print(json.dumps({"error": str(exc), "problems": exc.problems}, indent=2))
        else:
            print(f"get_models.py: {exc}", file=sys.stderr)
        return EXIT_INVALID
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())
