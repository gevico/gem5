# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import (
    ExitStack,
    contextmanager,
)
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
)

import psutil

SOURCE = Path(__file__).resolve().parents[2]
RAM_SIZE = 512 * 1024 * 1024
VRAM_SIZE = 1024 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def objects(value: Any, kind: str) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if value.get("type") == kind:
            yield value
        for child in value.values():
            yield from objects(child, kind)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child, kind)


def object_of_type(config: dict[str, Any], kind: str) -> dict[str, Any]:
    found = list(objects(config, kind))
    require(len(found) == 1, f"Expected one {kind}, found {len(found)}")
    return found[0]


def capture_output(
    stream: BinaryIO, log: BinaryIO, ready: threading.Event, listeners: int
) -> None:
    listening = 0
    for line in stream:
        log.write(line)
        if b"MI300XVfioUser: listening on " in line:
            listening += 1
            if listening == listeners:
                ready.set()


@contextmanager
def running_server(
    command: list[str], output: Path, listeners: int = 1
) -> Iterator[subprocess.Popen]:
    with (output / "output.log").open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            ready = threading.Event()
            capture = executor.submit(
                capture_output, process.stdout, log, ready, listeners
            )
            try:
                deadline = time.monotonic() + 120
                while not ready.is_set():
                    if capture.done():
                        capture.result()
                        raise RuntimeError(
                            f"MI300X output ended before readiness: {output}"
                        )
                    require(
                        process.poll() is None,
                        f"MI300X exited before readiness; see {output}",
                    )
                    require(
                        time.monotonic() < deadline,
                        f"MI300X readiness timed out; see {output}",
                    )
                    time.sleep(0.05)
                yield process
                require(
                    process.wait(timeout=30) == 0,
                    f"MI300X client disconnect failed; see {output}",
                )
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                capture.result(timeout=30)
                process.stdout.close()


def check_bar(
    client: Path, endpoint: Path, pool: Path, offset: int, output: Path
) -> None:
    require(endpoint.is_socket(), "vfio-user socket is unavailable")
    with (output / "vram-client.log").open("wb") as log:
        subprocess.run(
            [
                str(client),
                str(endpoint),
                str(pool),
                str(offset),
                str(VRAM_SIZE),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=True,
        )


def startup(
    binary: Path,
    client: Path,
    output: Path,
    start: int,
    ram_offset: int = 0,
    split: bool = False,
    vram_offset: int | None = None,
) -> None:
    output.mkdir()
    marker = b"MI300X pool ownership guard"
    page_size = os.sysconf("SC_PAGE_SIZE")
    if vram_offset is None:
        vram_offset = ram_offset + RAM_SIZE
    end = max(ram_offset + RAM_SIZE, vram_offset + VRAM_SIZE)
    pool_size = end + page_size
    guards = [end]
    if min(ram_offset, vram_offset) >= page_size:
        guards.append(0)
    with tempfile.TemporaryDirectory(prefix="mi3-", dir=output.parent) as tmp:
        pool = Path(tmp) / "pool"
        endpoint = Path(tmp) / "vfio.sock"
        with pool.open("xb") as stream:
            stream.truncate(pool_size)
            for offset in guards:
                stream.seek(offset)
                stream.write(marker)
        ranges = [{"start": start, "size": RAM_SIZE, "offset": ram_offset}]
        if split:
            ranges[0]["size"] = RAM_SIZE // 2
            ranges.append(
                {
                    "start": start + RAM_SIZE,
                    "size": RAM_SIZE // 2,
                    "offset": ram_offset + RAM_SIZE // 2,
                }
            )
        command = [
            str(binary),
            f"--outdir={output}",
            "--listener-mode=off",
            str(SOURCE / "configs/example/gpufs/mi300_cosim.py"),
            f"--socket-path={endpoint}",
            f"--shared-pool={pool}",
            f"--pool-vram-offset={vram_offset}",
            f"--mem-size={RAM_SIZE}B",
            f"--dgpu-mem-size={VRAM_SIZE}B",
            f"--host-memory-ranges={json.dumps(ranges)}",
            "--num-compute-units=4",
        ]
        with running_server(command, output) as process:
            mappings = [
                mapping
                for mapping in psutil.Process(process.pid).memory_maps(
                    grouped=False
                )
                if mapping.path == str(pool)
            ]
            require(
                len(mappings) == len(ranges) + 1,
                "Guest RAM and GPU VRAM must each be mapped once",
            )
            check_bar(client, endpoint, pool, vram_offset, output)

        log_text = (output / "output.log").read_text()
        require(
            "MI300XVfioUser: listening on " in log_text,
            f"MI300X did not finish initialization; see {output}",
        )
        require(pool.stat().st_size == pool_size, "MI300X resized the pool")
        with pool.open("rb") as stream:
            for offset in guards:
                stream.seek(offset)
                require(
                    stream.read(len(marker)) == marker,
                    f"Pool guard changed at offset {offset}",
                )

        config = json.loads((output / "config.json").read_text())
        system = config["system"]
        require(
            system["mem_ranges"]
            == [f"{r['start']}:{r['start'] + r['size']}" for r in ranges],
            "Guest physical range differs from the supplied map",
        )
        require(
            system["shared_backstore"] == str(pool)
            and system["shared_backstore_offset"] == ram_offset
            and system["shared_backstore_limit"] == RAM_SIZE,
            "Host RAM does not use the assigned pool region",
        )
        gpu = object_of_type(config, "AMDGPUDevice")
        require(
            gpu["vram_shared_backstore"] == str(pool)
            and gpu["vram_shared_backstore_offset"] == vram_offset
            and gpu["vram_shared_backstore_limit"] == VRAM_SIZE,
            "MI300X VRAM does not use the assigned pool region",
        )


def build_client(binary: Path, output: Path) -> Path:
    client = output / "vram-client"
    library = SOURCE / "ext/libvfio-user"
    command = [
        os.environ.get("CC", "cc"),
        "-std=gnu11",
        "-D_GNU_SOURCE",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{library / 'include'}",
        f"-I{library / 'lib'}",
        str(SOURCE / "tests/mi300x/vram_client.c"),
        str(binary.parent / "ext/libvfio-user/libvfio-user.a"),
        "-ljson-c",
        "-pthread",
        "-o",
        str(client),
    ]
    with (output / "vram-client-build.log").open("wb") as log:
        subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=True,
        )
    return client


def named_multi_gpu(binary: Path, client: Path, output: Path) -> None:
    output.mkdir()
    name = f"mi300x-{uuid.uuid4().hex}"
    pools = [Path("/dev/shm") / f"{name}-{index}" for index in range(2)]
    ram = Path("/dev/shm") / f"{name}-ram"
    endpoint = output / "vfio.sock"
    command = [
        str(binary),
        f"--outdir={output}",
        "--listener-mode=off",
        str(SOURCE / "configs/example/gpufs/mi300_cosim.py"),
        f"--socket-path={endpoint}",
        f"--shmem-path=/{name}",
        f"--shmem-host-path=/{name}-ram",
        f"--mem-size={RAM_SIZE}B",
        f"--dgpu-mem-size={VRAM_SIZE}B",
        "--num-gpus=2",
        "--num-compute-units=4",
    ]
    try:
        with ExitStack() as files:
            with running_server(command, output, listeners=2) as process:
                mappings = psutil.Process(process.pid).memory_maps(
                    grouped=False
                )
                for pool in pools:
                    require(
                        sum(mapping.path == str(pool) for mapping in mappings)
                        == 1,
                        "Each GPU must own one VRAM mapping",
                    )
                stream = files.enter_context(pools[0].open("rb"))
                check_bar(client, output / "vfio-1.sock", pools[1], 0, output)
            for offset in [4096, 8192]:
                stream.seek(offset)
                require(
                    stream.read(32) == bytes(32),
                    "Accessing GPU 1 changed GPU 0 VRAM",
                )
            for path in [ram, *pools]:
                require(
                    not path.exists(),
                    f"Owned POSIX memory remains after normal exit: {path}",
                )
    finally:
        for path in [ram, *pools]:
            path.unlink(missing_ok=True)


def invalid_ranges(binary: Path, output: Path) -> None:
    output.mkdir()
    page = os.sysconf("SC_PAGE_SIZE")
    valid = [{"start": 0, "size": RAM_SIZE, "offset": page}]
    cases = {
        "overlap": (valid, RAM_SIZE),
        "negative": ([{**valid[0], "offset": -page}], RAM_SIZE + page),
        "unaligned": ([{**valid[0], "offset": 1}], RAM_SIZE + page),
        "overflow": ([{**valid[0], "offset": 1 << 63}], RAM_SIZE + page),
        "size": ([{**valid[0], "size": RAM_SIZE // 2}], RAM_SIZE + page),
        "unpacked": (
            [
                {**valid[0], "size": RAM_SIZE // 2},
                {"start": RAM_SIZE, "size": RAM_SIZE // 2, "offset": RAM_SIZE},
            ],
            RAM_SIZE + page,
        ),
    }
    for name, (ranges, vram_offset) in cases.items():
        command = [
            str(binary),
            f"--outdir={output / name}",
            "--listener-mode=off",
            str(SOURCE / "configs/example/gpufs/mi300_cosim.py"),
            f"--shared-pool={output / 'pool'}",
            f"--pool-vram-offset={vram_offset}",
            f"--mem-size={RAM_SIZE}B",
            f"--host-memory-ranges={json.dumps(ranges)}",
        ]
        with (output / f"{name}.log").open("wb") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
        text = (output / f"{name}.log").read_text()
        require(result.returncode != 0, f"Accepted invalid memory map: {name}")
        require(
            "host memory" in text or "Pool VRAM overlaps guest RAM" in text,
            f"Unexpected rejection for {name}; see {output}",
        )


def protocol(binary: Path, output: Path) -> None:
    output.mkdir()
    command = [
        str(binary),
        f"--outdir={output}",
        "--listener-mode=off",
        str(SOURCE / "configs/example/ruby_gpu_random_test.py"),
        "--test-length=50000",
        "--num-dmas=0",
        "--random-seed=1",
        f"--log-file={output / 'tester.log'}",
    ]
    with (output / "output.log").open("wb") as log:
        subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
            check=True,
        )
    require(
        "GPU Ruby Tester: Passed!" in (output / "output.log").read_text(),
        f"GPU_VIPER completion was not reported; see {output}",
    )
    config = json.loads((output / "config.json").read_text())
    object_of_type(config, "GPU_VIPER_TCP_Controller")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gem5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case", choices=["all", "named-multi-gpu", "mmhub"], default="all"
    )
    args = parser.parse_args()
    binary = args.gem5.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {"binary": str(binary), "case": args.case, "status": "FAIL"}
    try:
        client = build_client(binary, output)
        if args.case == "named-multi-gpu":
            named_multi_gpu(binary, client, output / "named-multi-gpu")
            result["status"] = "PASS"
            return
        if args.case == "mmhub":
            startup(binary, client, output / "mmhub", 0)
            result["status"] = "PASS"
            return
        startup(binary, client, output / "q35", 0)
        startup(binary, client, output / "aarch64", 0x40000000)
        page = os.sysconf("SC_PAGE_SIZE")
        startup(binary, client, output / "offset-split", 0, page, split=True)
        startup(
            binary,
            client,
            output / "ram-after-vram",
            0x40000000,
            page + VRAM_SIZE,
            vram_offset=page,
        )
        named_multi_gpu(binary, client, output / "named-multi-gpu")
        invalid_ranges(binary, output / "invalid-ranges")
        protocol(binary, output / "gpu-viper")
        result["status"] = "PASS"
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        (output / "verdict.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
