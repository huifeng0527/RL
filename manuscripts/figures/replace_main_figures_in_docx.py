from __future__ import annotations

import re
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPTS = ROOT / "manuscripts"
PAPER_READY = MANUSCRIPTS / "figures" / "paper_ready"
INPUT_DOCX = MANUSCRIPTS / "manuscripts_methodology_v2.88_results_concise_table2.docx"
OUTPUT_DOCX = MANUSCRIPTS / "manuscripts_methodology_v2.89_figures_standardized.docx"
DOCUMENT_XML = "word/document.xml"
DISPLAY_WIDTH_EMU = 6_035_040

REPLACEMENTS = {
    "word/media/image3.png": PAPER_READY / "fig03_league_overview_labels_only_v2.89.png",
    "word/media/image4.png": PAPER_READY / "fig04_network_ablation_labels_only_v2.89.png",
    "word/media/image5.png": PAPER_READY / "fig05_physical_deployment_labels_only_v2.89.png",
}

RID_TO_MEDIA = {
    "rId8": "word/media/image1.png",
    "rId9": "word/media/image2.png",
    "rId10": "word/media/image3.png",
    "rId11": "word/media/image4.png",
    "rId12": "word/media/image5.png",
}

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def image_size_from_bytes(data: bytes) -> tuple[int, int]:
    from io import BytesIO

    with Image.open(BytesIO(data)) as image:
        return image.size


def replace_extent(tag: bytes, cx: int, cy: int, prefix: bytes) -> bytes:
    pattern = re.compile(
        rb"(<" + prefix + rb':ext(?:ent)?\b[^>]*\bcx=")(\d+)("[^>]*\bcy=")(\d+)(")'
    )

    def replacement(match: re.Match[bytes]) -> bytes:
        return match.group(1) + str(cx).encode() + match.group(3) + str(cy).encode() + match.group(5)

    updated, count = pattern.subn(replacement, tag, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one {prefix.decode()} extent, found {count}")
    return updated


def update_drawing_extent(xml: bytes, rid: str, cx: int, cy: int) -> bytes:
    inline_pattern = re.compile(rb"<wp:inline\b.*?</wp:inline>", re.DOTALL)
    rid_marker = f'r:embed="{rid}"'.encode()
    matches = [match for match in inline_pattern.finditer(xml) if rid_marker in match.group(0)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one inline drawing for {rid}, found {len(matches)}")

    match = matches[0]
    block = replace_extent(match.group(0), cx, cy, b"wp")
    block = replace_extent(block, cx, cy, b"a")
    return xml[: match.start()] + block + xml[match.end() :]


def text_sequence(xml: bytes) -> list[str]:
    root = etree.fromstring(xml)
    return root.xpath("//w:t/text()", namespaces=NS)


def main() -> None:
    if not INPUT_DOCX.exists():
        raise FileNotFoundError(INPUT_DOCX)
    for path in REPLACEMENTS.values():
        if not path.exists():
            raise FileNotFoundError(path)

    with zipfile.ZipFile(INPUT_DOCX, "r") as source:
        infos = source.infolist()
        original_members = {info.filename: source.read(info.filename) for info in infos}

    output_members = dict(original_members)
    output_members.update({member: path.read_bytes() for member, path in REPLACEMENTS.items()})

    document_xml = original_members[DOCUMENT_XML]
    for rid, media_member in RID_TO_MEDIA.items():
        width_px, height_px = image_size_from_bytes(output_members[media_member])
        height_emu = round(DISPLAY_WIDTH_EMU * height_px / width_px)
        document_xml = update_drawing_extent(document_xml, rid, DISPLAY_WIDTH_EMU, height_emu)
        print(f"{rid}: {width_px}x{height_px} -> {DISPLAY_WIDTH_EMU}x{height_emu} EMU")
    output_members[DOCUMENT_XML] = document_xml

    if text_sequence(original_members[DOCUMENT_XML]) != text_sequence(document_xml):
        raise RuntimeError("Manuscript text changed during figure replacement")
    if output_members["word/_rels/document.xml.rels"] != original_members["word/_rels/document.xml.rels"]:
        raise RuntimeError("Document relationships changed")
    if output_members["[Content_Types].xml"] != original_members["[Content_Types].xml"]:
        raise RuntimeError("Content types changed")

    with zipfile.ZipFile(OUTPUT_DOCX, "w") as target:
        for info in infos:
            target.writestr(info, output_members[info.filename])

    print(OUTPUT_DOCX)


if __name__ == "__main__":
    main()
