from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PAPER_READY = ROOT / "manuscripts" / "figures" / "paper_ready"
FONT_PATH = Path("C:/Windows/Fonts/arialbd.ttf")
FINAL_WIDTH_IN = 6.6
LABEL_SIZE_PT = 9.0
WHITE = (255, 255, 255)


@dataclass(frozen=True)
class FigureSpec:
    source: Path
    output: Path
    crop_top: int
    masks: tuple[tuple[int, int, int, int], ...]
    labels: tuple[tuple[str, int, int], ...]


def label_font(image_width: int) -> ImageFont.FreeTypeFont:
    size_px = round(image_width * LABEL_SIZE_PT / (FINAL_WIDTH_IN * 72.0))
    return ImageFont.truetype(str(FONT_PATH), size=max(size_px, 1))


def shifted_box(box: tuple[int, int, int, int], crop_top: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return left, top - crop_top, right, bottom - crop_top


def render(spec: FigureSpec) -> None:
    source = Image.open(spec.source).convert("RGB")
    dpi = source.info.get("dpi", (300, 300))
    base = source.crop((0, spec.crop_top, source.width, source.height))
    result = base.copy()
    draw = ImageDraw.Draw(result)
    allowed = Image.new("L", result.size, 0)
    allowed_draw = ImageDraw.Draw(allowed)

    for raw_box in spec.masks:
        box = shifted_box(raw_box, spec.crop_top)
        draw.rectangle(box, fill=WHITE)
        allowed_draw.rectangle(box, fill=255)

    font = label_font(result.width)
    for text, raw_x, raw_y in spec.labels:
        x = raw_x
        y = raw_y - spec.crop_top
        draw.text((x, y), text, font=font, fill=(0, 0, 0), anchor="la")
        bbox = draw.textbbox((x, y), text, font=font, anchor="la")
        allowed_draw.rectangle(bbox, fill=255)

    difference = ImageChops.difference(base, result).convert("L")
    outside = ImageChops.multiply(difference, ImageChops.invert(allowed))
    if outside.getbbox() is not None:
        raise RuntimeError(f"Unexpected pixel changes outside title/label regions for {spec.source}")

    spec.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(spec.output, dpi=dpi, optimize=True)
    print(f"{spec.output} ({result.width}x{result.height})")


def main() -> None:
    specs = (
        FigureSpec(
            source=PAPER_READY / "fig_sim01_league_overview_no_title.png",
            output=PAPER_READY / "fig03_league_overview_labels_only_v2.89.png",
            crop_top=0,
            masks=((0, 0, 2746, 116),),
            labels=(("(a)", 66, 18), ("(b)", 1380, 18)),
        ),
        FigureSpec(
            source=PAPER_READY / "fig_sim02_ablation_gru_aux_composite_filled_no_title.png",
            output=PAPER_READY / "fig04_network_ablation_labels_only_v2.89.png",
            crop_top=0,
            masks=(
                (0, 0, 3590, 88),
                (0, 1115, 2050, 1235),
                (2050, 1160, 3590, 1290),
            ),
            labels=(
                ("(a)", 120, 10),
                ("(b)", 2110, 10),
                ("(c)", 120, 1135),
                ("(d)", 2110, 1180),
            ),
        ),
        FigureSpec(
            source=PAPER_READY / "fig_physical_deployment.png",
            output=PAPER_READY / "fig05_physical_deployment_labels_only_v2.89.png",
            crop_top=110,
            masks=((0, 150, 3327, 260), (0, 740, 3327, 865)),
            labels=(("(a)", 120, 175), ("(b)", 120, 770)),
        ),
    )

    for spec in specs:
        render(spec)


if __name__ == "__main__":
    main()
