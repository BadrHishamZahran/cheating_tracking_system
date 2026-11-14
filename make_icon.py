"""Generates assets/icon.ico and assets/icon.png. Replace them with your own artwork any time."""
from pathlib import Path
from PIL import Image, ImageDraw

out = Path(__file__).parent / "assets"
out.mkdir(exist_ok=True)
S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((8, 8, S - 8, S - 8), radius=48, fill=(24, 32, 56, 255))
d.ellipse((48, 70, S - 48, S - 70), outline=(255, 255, 255, 255), width=12)   # eye
d.ellipse((98, 98, S - 98, S - 98), fill=(220, 38, 38, 255))                  # red pupil
d.ellipse((118, 108, 138, 128), fill=(255, 255, 255, 255))
img.save(out / "icon.png")
img.save(out / "icon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("icon created")
