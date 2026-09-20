import json
import os
import re
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

BASE = Path(os.getenv("COROLLA_DATA", "/data"))
INPUTS = BASE / "inputs"
JOBS = BASE / "jobs"

INPUTS.mkdir(parents=True, exist_ok=True)
JOBS.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="CorollaCloud",
    version="0.1.0"
)

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


def run_cmd(
    args: list[str],
    timeout: int = 3600
) -> tuple[int, str, str]:

    process = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout
    )

    return (
        process.returncode,
        process.stdout,
        process.stderr
    )


def safe_name(name: str) -> str:

    name = Path(
        name or "video.mp4"
    ).name

    name = re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        name
    )

    return name or "video.mp4"


def ffprobe_json(
    path: Path,
    frames: bool = False
) -> dict[str, Any]:

    if frames:

        args = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=pict_type,key_frame",
            str(path)
        ]

    else:

        args = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path)
        ]

    rc, out, err = run_cmd(
        args,
        timeout=1800
    )

    if rc != 0:

        raise RuntimeError(
            err[-4000:]
            or "ffprobe failed"
        )

    return json.loads(
        out or "{}"
    )


def stream_summary(
    path: Path
) -> dict[str, Any]:

    data = ffprobe_json(path)

    streams = data.get(
        "streams",
        []
    )

    video = next(
        (
            s for s in streams
            if s.get("codec_type") == "video"
        ),
        {}
    )

    audio = next(
        (
            s for s in streams
            if s.get("codec_type") == "audio"
        ),
        {}
    )

    fmt = data.get(
        "format",
        {}
    )

    def num(
        obj: dict[str, Any],
        key: str
    ):

        value = obj.get(key)

        if str(value).isdigit():
            return int(value)

        return value

    return {

        "codec":
            video.get("codec_name"),

        "profile":
            video.get("profile"),

        "tag":
            video.get("codec_tag_string"),

        "width":
            video.get("width"),

        "height":
            video.get("height"),

        "pix_fmt":
            video.get("pix_fmt"),

        "fps":
            video.get("r_frame_rate"),

        "avg_fps":
            video.get("avg_frame_rate"),

        "time_base":
            video.get("time_base"),

        "video_bitrate":
            num(
                video,
                "bit_rate"
            ),

        "has_b_frames":
            video.get("has_b_frames"),

        "refs":
            video.get("refs"),

        "level":
            video.get("level"),

        "colorspace":
            video.get("color_space"),

        "primaries":
            video.get("color_primaries"),

        "transfer":
            video.get("color_transfer"),

        "range":
            video.get("color_range"),

        "audio_codec":
            audio.get("codec_name"),

        "audio_bitrate":
            num(
                audio,
                "bit_rate"
            ),

        "format":
            fmt.get("format_name"),

        "format_bitrate":
            num(
                fmt,
                "bit_rate"
            ),

        "duration":
            fmt.get("duration"),

        "size":
            num(
                fmt,
                "size"
            )
    }


def frame_summary(
    path: Path
) -> dict[str, Any]:

    data = ffprobe_json(
        path,
        frames=True
    )

    frames = data.get(
        "frames",
        []
    )

    counts = {
        "I": 0,
        "P": 0,
        "B": 0,
        "other": 0
    }

    max_run = 0
    current_run = 0

    positions = []
    pattern = []

    for idx, frame in enumerate(
        frames
    ):

        pict_type = frame.get(
            "pict_type"
        )

        if pict_type in (
            "I",
            "P",
            "B"
        ):

            counts[pict_type] += 1

            pattern.append(
                pict_type
            )

            if pict_type == "B":

                current_run += 1

                max_run = max(
                    max_run,
                    current_run
                )

            else:

                current_run = 0

            if pict_type == "I":

                positions.append(
                    idx
                )

        else:

            counts["other"] += 1

            pattern.append("?")

            current_run = 0

    return {

        "frames":
            len(frames),

        **counts,

        "max_b_run":
            max_run,

        "i_positions":
            positions[:100],

        "pattern":
            " ".join(
                pattern[:100]
            )
            +
            (
                " ..."
                if len(pattern) > 100
                else ""
            )
    }


class Experiment(BaseModel):

    encoder: str = Field(
        default="libx265"
    )

    bitrate: str = Field(
        default="733k"
    )

    duration: int = Field(
        default=6,
        ge=1,
        le=120
    )

    bframes: int = Field(
        default=4,
        ge=0,
        le=16
    )

    b_adapt: int = Field(
        default=0,
        ge=0,
        le=2
    )

    b_pyramid: int = Field(
        default=0,
        ge=0,
        le=1
    )

    ref: int = Field(
        default=1,
        ge=1,
        le=8
    )

    keyint: int = Field(
        default=184,
        ge=1,
        le=1000
    )

    scenecut: int = Field(
        default=0,
        ge=0,
        le=1
    )


def build_x265_command(
    src: Path,
    out: Path,
    cfg: Experiment
) -> list[str]:

    params = (
        f"bframes={cfg.bframes}:"
        f"b-adapt={cfg.b_adapt}:"
        f"b-pyramid="
        f"{'normal' if cfg.b_pyramid else 'none'}:"
        f"ref={cfg.ref}:"
        f"keyint={cfg.keyint}:"
        f"min-keyint={cfg.keyint}:"
        f"scenecut={cfg.scenecut}"
    )

    return [

        "ffmpeg",
        "-y",
        "-v",
        "error",

        "-i",
        str(src),

        "-t",
        str(cfg.duration),

        "-map",
        "0:v:0",

        "-an",

        "-vf",
        (
            "scale=1080:1920:"
            "force_original_aspect_ratio=decrease,"
            "pad=1080:1920:"
            "(ow-iw)/2:(oh-ih)/2,"
            "format=yuv420p,"
            "setsar=1"
        ),

        "-r",
        "30",

        "-c:v",
        cfg.encoder,

        "-pix_fmt",
        "yuv420p",

        "-profile:v",
        "main",

        "-x265-params",
        params,

        "-b:v",
        cfg.bitrate,

        "-tag:v",
        "hvc1",

        "-colorspace",
        "bt709",

        "-color_primaries",
        "bt709",

        "-color_trc",
        "bt709",

        str(out)
    ]


def do_experiment(
    job_id: str,
    src: Path,
    cfg: Experiment
):

    out = JOBS / job_id / "result.mp4"

    try:

        with jobs_lock:

            jobs[job_id].update(
                status="running",
                message="Encoding..."
            )

        if cfg.encoder != "libx265":

            raise RuntimeError(
                "v0.1 currently supports "
                "encoder=libx265 only"
            )

        rc, _, stderr = run_cmd(

            build_x265_command(
                src,
                out,
                cfg
            ),

            timeout=max(
                1800,
                cfg.duration * 180
            )
        )

        if rc != 0:

            raise RuntimeError(
                stderr[-8000:]
                or "FFmpeg failed"
            )

        with jobs_lock:

            jobs[job_id][
                "message"
            ] = "Analyzing metadata..."

        metadata = stream_summary(
            out
        )

        frames = frame_summary(
            out
        )

        with jobs_lock:

            jobs[job_id].update(

                status="done",

                message="Complete",

                metadata=metadata,

                frame_structure=frames,

                result_url=(
                    f"/api/jobs/"
                    f"{job_id}/download"
                )
            )

    except Exception as exc:

        with jobs_lock:

            jobs[job_id].update(
                status="error",
                message=str(exc)
            )


@app.get(
    "/",
    response_class=HTMLResponse
)
def index():

    return (
        Path(__file__).parent
        / "static"
        / "index.html"
    ).read_text(
        encoding="utf-8"
    )


@app.get("/api/health")
def health():

    rc, out, err = run_cmd(
        [
            "ffmpeg",
            "-hide_banner",
            "-version"
        ],
        timeout=30
    )

    enc_rc, enc_out, enc_err = run_cmd(
        [
            "ffmpeg",
            "-hide_banner",
            "-encoders"
        ],
        timeout=30
    )

    return {

        "ok":
            rc == 0,

        "ffmpeg":
            (
                out.splitlines()[0]
                if out
                else err
            ),

        "encoders":
            [
                line.strip()

                for line
                in enc_out.splitlines()

                if (
                    "hevc"
                    in line.lower()
                    or
                    "x265"
                    in line.lower()
                )
            ]
    }


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            400,
            "No filename"
        )

    file_id = uuid.uuid4().hex

    destination = (
        INPUTS
        /
        f"{file_id}_"
        f"{safe_name(file.filename)}"
    )

    with destination.open(
        "wb"
    ) as output:

        while chunk := await file.read(
            1024 * 1024
        ):

            output.write(chunk)

    return {

        "file_id":
            file_id,

        "filename":
            destination.name,

        "size":
            destination.stat().st_size
    }


@app.post(
    "/api/experiment/{file_id}"
)
def experiment(
    file_id: str,
    cfg: Experiment
):

    matches = list(
        INPUTS.glob(
            f"{file_id}_*"
        )
    )

    if not matches:

        raise HTTPException(
            404,
            "Input file not found"
        )

    job_id = uuid.uuid4().hex

    work = (
        JOBS
        /
        job_id
    )

    work.mkdir(
        parents=True,
        exist_ok=True
    )

    with jobs_lock:

        jobs[job_id] = {

            "job_id":
                job_id,

            "status":
                "queued",

            "message":
                "Queued",

            "config":
                cfg.model_dump()
        }

    threading.Thread(

        target=do_experiment,

        args=(
            job_id,
            matches[0],
            cfg
        ),

        daemon=True
    ).start()

    return jobs[job_id]


@app.get(
    "/api/jobs/{job_id}"
)
def job_status(
    job_id: str
):

    with jobs_lock:

        item = jobs.get(
            job_id
        )

        if not item:

            raise HTTPException(
                404,
                "Job not found"
            )

        return item


@app.get(
    "/api/jobs/{job_id}/download"
)
def download(
    job_id: str
):

    path = (
        JOBS
        /
        job_id
        /
        "result.mp4"
    )

    if not path.exists():

        raise HTTPException(
            404,
            "Result not ready"
        )

    return FileResponse(
        path,
        media_type="video/mp4",
        filename="corolla_result.mp4"
              )
