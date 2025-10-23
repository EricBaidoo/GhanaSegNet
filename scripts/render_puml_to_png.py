"""
Simple renderer: convert .puml files to PNGs by drawing the PUML source text onto an image.
This is a placeholder renderer (shows the PUML text) and does not produce true UML diagrams.
Run: python scripts\render_puml_to_png.py
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DIAGRAMS_DIR = Path(__file__).resolve().parents[1] / 'diagrams'
FIGURES_DIR = Path(__file__).resolve().parents[1] / 'figures'

FIGURES_DIR.mkdir(exist_ok=True)

font = ImageFont.load_default()

puml_files = sorted(DIAGRAMS_DIR.glob('*.puml'))
if not puml_files:
    print('No .puml files found in', DIAGRAMS_DIR)
    raise SystemExit(0)

for p in puml_files:
    text = p.read_text(encoding='utf-8')
    lines = text.splitlines()
    # measure using a temporary ImageDraw
    tmp_img = Image.new('RGB', (10, 10))
    tmp_draw = ImageDraw.Draw(tmp_img)
    def measure(text):
        # Try textbbox (newer Pillow), else fallback to font.getmask
        try:
            bbox = tmp_draw.textbbox((0,0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            return w, h
        except Exception:
            mask = font.getmask(text)
            return mask.size

    max_width = max((measure(line)[0] for line in lines), default=200)
    line_height = measure('A')[1]
    padding = 16
    img_w = max_width + padding*2
    img_h = line_height * len(lines) + padding*2

    # Create white background image
    img = Image.new('RGB', (img_w, img_h), color=(255,255,255))
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill=(0,0,0))
        y += line_height

    out_path = FIGURES_DIR / (p.stem + '.png')
    img.save(out_path, format='PNG')
    print('Wrote', out_path)

print('Done. Generated', len(puml_files), 'PNG(s) in', FIGURES_DIR)
