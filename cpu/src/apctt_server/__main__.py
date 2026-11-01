"""Start the server: python -m apctt_server --profile dev --config config/dev.json

One Uvicorn worker only; no auto-reload. The profile must match the configuration file.
Startup is refused with a fixed problem code when a gate in SPEC-03/12 is not met.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from apctt_contracts.config import ConfigError, load_config, startup_problems
from apctt_server.app import create_app
from apctt_server.store import data_dir_problem

EXIT_CONFIG = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="apctt_server", description=__doc__)
    parser.add_argument("--profile", choices=("dev", "evaluation"), default="dev")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        loaded = load_config(args.config, profile=args.profile)
    except ConfigError as exc:
        print(f"APCTT_START_REFUSED {exc.code}", file=sys.stderr)
        for detail in exc.details:
            print(f"  {detail}", file=sys.stderr)
        return EXIT_CONFIG
    problems = startup_problems(loaded)
    data_dir = data_dir_problem(loaded.data_dir)
    if data_dir is not None:
        problems.append(data_dir)
    if problems:
        print("APCTT_START_REFUSED " + " ".join(problems), file=sys.stderr)
        return EXIT_CONFIG
    config = loaded.config
    uvicorn.run(
        create_app(loaded),
        host=config.bind_host,
        port=config.bind_port,
        workers=1,
        proxy_headers=False,
        server_header=False,
        access_log=False,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
