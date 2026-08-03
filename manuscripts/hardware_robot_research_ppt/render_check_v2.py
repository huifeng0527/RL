"""Render v2 deck to PNG, build a contact sheet, and check for text overflow."""
import shutil
from pathlib import Path

import win32com.client
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
PPT = BASE / "康复桌面抓取目标微型移动机器人_运动机理分类调研_汇报版_v3_ESPROLL.pptx"
OUTDIR = BASE / "renders_v3_ESPROLL"

if OUTDIR.exists():
    shutil.rmtree(OUTDIR)
OUTDIR.mkdir(parents=True)

app = win32com.client.DispatchEx("PowerPoint.Application")
pres = app.Presentations.Open(str(PPT), WithWindow=False)

overflow = []
for slide in pres.Slides:
    for shp in slide.Shapes:
        if not shp.HasTextFrame:
            continue
        tf2 = shp.TextFrame2
        if not tf2.HasText:
            continue
        bh = tf2.TextRange.BoundHeight
        bw = tf2.TextRange.BoundWidth
        if bh > shp.Height + 1 or bw > shp.Width + 1:
            overflow.append(
                (slide.SlideIndex, shp.Name, round(bw, 1), round(shp.Width, 1),
                 round(bh, 1), round(shp.Height, 1),
                 tf2.TextRange.Text[:40].replace("\r", " "))
            )

pres.Export(str(OUTDIR), "PNG", 1600, 900)
pres.Close()
app.Quit()

print("overflow_candidates", len(overflow))
for row in overflow:
    print("  ", row)

pngs = sorted(OUTDIR.glob("*.PNG")) + sorted(OUTDIR.glob("*.png"))
pngs = sorted(set(pngs), key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
print("rendered", len(pngs))

cols, tw = 3, 620
rows = (len(pngs) + cols - 1) // cols
th = int(tw * 9 / 16)
sheet = Image.new("RGB", (cols * tw + (cols + 1) * 12, rows * (th + 26) + 12), "white")
d = ImageDraw.Draw(sheet)
for i, p in enumerate(pngs):
    im = Image.open(p).convert("RGB").resize((tw, th))
    x = 12 + (i % cols) * (tw + 12)
    y = 12 + (i // cols) * (th + 26)
    sheet.paste(im, (x, y))
    d.rectangle([x, y, x + tw, y + th], outline=(200, 200, 200))
    d.text((x + 2, y + th + 6), f"{i+1:02d}  {p.name}", fill=(60, 60, 60))
sheet.save(OUTDIR / "contact_sheet.jpg", quality=88)
print(OUTDIR / "contact_sheet.jpg")
