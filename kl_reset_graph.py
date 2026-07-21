#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plot reference-reset evidence extracted from a verl console log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"\bstep:(\d+)\s+-")
RESET_RE = re.compile(r"\btraining/ref_policy_reset:(\d+)")
KL_RE = re.compile(
    r"\bactor/kl_loss:(?:np\.float(?:32|64)\()?"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_metrics(log_path: Path) -> list[tuple[int, int, float]]:
    """Return (step, reset event, actor KL loss) records from a console log."""
    records: dict[int, tuple[int, float]] = {}
    text = ANSI_ESCAPE_RE.sub("", log_path.read_text(encoding="utf-8", errors="replace"))

    for line in text.splitlines():
        step_match = STEP_RE.search(line)
        reset_match = RESET_RE.search(line)
        kl_match = KL_RE.search(line)
        if step_match and reset_match and kl_match:
            step = int(step_match.group(1))
            records[step] = (int(reset_match.group(1)), float(kl_match.group(1)))

    if not records:
        raise ValueError(
            f"No records containing step, training/ref_policy_reset, and actor/kl_loss found in {log_path}"
        )

    return [(step, *records[step]) for step in sorted(records)]


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def dashed_vertical(draw: ImageDraw.ImageDraw, x: int, top: int, bottom: int, fill: str, width: int = 3) -> None:
    for y in range(top, bottom, 18):
        draw.line((x, y, x, min(y + 10, bottom)), fill=fill, width=width)


def render_graph(records: list[tuple[int, int, float]], output_path: Path) -> None:
    width, height = 1600, 1000
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)

    title_font = load_font(42, bold=True)
    subtitle_font = load_font(23)
    label_font = load_font(22)
    small_font = load_font(18)
    small_bold_font = load_font(18, bold=True)

    draw.text((80, 45), "Reference-policy reset evidence", font=title_font, fill="#172033")
    draw.text(
        (82, 101),
        "actor/kl_loss drops on the step following each reference reset",
        font=subtitle_font,
        fill="#526079",
    )

    left, right, top, bottom = 155, 1520, 185, 700
    steps = [record[0] for record in records]
    resets = {step for step, reset, _ in records if reset == 1}
    values = {step: kl for step, _, kl in records}
    min_step, max_step = min(steps), max(steps)
    y_max = max(values.values()) * 1.18

    def x_pos(step: int) -> int:
        if min_step == max_step:
            return (left + right) // 2
        return round(left + (step - min_step) * (right - left) / (max_step - min_step))

    def y_pos(value: float) -> int:
        return round(bottom - value * (bottom - top) / y_max)

    # Horizontal grid and y-axis labels.
    for index in range(6):
        value = y_max * index / 5
        y = y_pos(value)
        draw.line((left, y, right, y), fill="#dfe4ec", width=2)
        draw.text(
            (left - 18, y),
            f"{value:.4f}",
            font=small_font,
            fill="#526079",
            anchor="rm",
        )

    draw.line((left, top, left, bottom), fill="#7b879b", width=3)
    draw.line((left, bottom, right, bottom), fill="#7b879b", width=3)
    draw.text((30, (top + bottom) // 2), "actor/kl_loss", font=label_font, fill="#172033", anchor="mm")

    # Scheduled resets happen after the KL value for that step has been computed.
    for step in sorted(resets):
        x = x_pos(step)
        dashed_vertical(draw, x, top, bottom, "#d1495b")
        draw.text((x, top - 12), f"reset after step {step}", font=small_bold_font, fill="#b1283d", anchor="mb")

    points = [(x_pos(step), y_pos(values[step])) for step in steps]
    if len(points) > 1:
        draw.line(points, fill="#2166ac", width=6, joint="curve")

    for step, (x, y) in zip(steps, points, strict=True):
        fill = "#d1495b" if step in resets else "#2166ac"
        if step - 1 in resets:
            fill = "#14866d"
        radius = 9
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline="white", width=3)
        label_y = y - 20 if step % 2 else y + 20
        anchor = "mb" if step % 2 else "mt"
        draw.text((x, label_y), f"{values[step]:.6f}", font=small_font, fill="#172033", anchor=anchor)
        draw.text((x, bottom + 18), str(step), font=small_font, fill="#526079", anchor="mt")

    draw.text(((left + right) // 2, bottom + 55), "Training step", font=label_font, fill="#172033", anchor="mt")

    # Annotate each reset-to-following-step decrease.
    for reset_step in sorted(resets):
        next_step = reset_step + 1
        if next_step not in values or values[reset_step] == 0:
            continue
        drop = 100 * (values[reset_step] - values[next_step]) / values[reset_step]
        x = (x_pos(reset_step) + x_pos(next_step)) // 2
        y = (y_pos(values[reset_step]) + y_pos(values[next_step])) // 2
        draw.rounded_rectangle((x - 65, y - 22, x + 65, y + 22), radius=10, fill="#e6f4ef", outline="#14866d")
        draw.text((x, y), f"{drop:.1f}% drop", font=small_bold_font, fill="#106c58", anchor="mm")

    # Reset-event panel makes the binary event metric explicit rather than relying only on annotations.
    event_top, event_bottom = 815, 895
    draw.text((left, event_top - 22), "training/ref_policy_reset", font=label_font, fill="#172033", anchor="ls")
    draw.line((left, event_bottom, right, event_bottom), fill="#aab3c2", width=3)
    for step, reset, _ in records:
        x = x_pos(step)
        event_y = event_top if reset else event_bottom
        draw.line((x, event_bottom, x, event_y), fill="#d1495b" if reset else "#aab3c2", width=6)
        radius = 8
        draw.ellipse(
            (x - radius, event_y - radius, x + radius, event_y + radius),
            fill="#d1495b" if reset else "#aab3c2",
        )
        draw.text((x, event_bottom + 17), str(step), font=small_font, fill="#526079", anchor="mt")
    draw.text((left - 18, event_top), "1", font=small_font, fill="#526079", anchor="rm")
    draw.text((left - 18, event_bottom), "0", font=small_font, fill="#526079", anchor="rm")

    draw.text(
        (width - 80, height - 38),
        "Source: verl console training metrics",
        font=small_font,
        fill="#7b879b",
        anchor="rs",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("e2e_ref_reset.txt"),
        help="verl console log (default: e2e_ref_reset.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output PNG path (default: input path with a .png suffix)",
    )
    args = parser.parse_args()

    output_path = args.output or args.input.with_suffix(".png")
    records = parse_metrics(args.input)
    render_graph(records, output_path)
    print(f"Wrote {output_path} from {len(records)} training steps.")


if __name__ == "__main__":
    main()
