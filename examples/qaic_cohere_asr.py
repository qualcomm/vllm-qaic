# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Transcribe one audio file through a running Cohere ASR vLLM server.

Start the server with the Cohere ASR QPC configuration documented in
``docs/docs/user_guide/features/encoder_decoder.md``, then run:

    python examples/qaic_cohere_asr.py --file-path /path/to/audio.wav --language en
"""

import argparse
from pathlib import Path

from openai import OpenAI


DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file-path", type=Path, required=True, help="Audio file to transcribe"
    )
    parser.add_argument(
        "--language",
        required=True,
        help=(
            "BCP-47 language code passed to the transcription endpoint, "
            "for example en or ar"
        ),
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Served Cohere ASR model ID"
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="vLLM OpenAI API base URL"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.file_path.is_file():
        raise ValueError(f"Audio file does not exist: {args.file_path}")

    client = OpenAI(api_key="EMPTY", base_url=args.base_url)
    with args.file_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model=args.model,
            language=args.language,
            response_format="json",
            temperature=0.0,
        )
    print(transcription.text)


if __name__ == "__main__":
    main()
