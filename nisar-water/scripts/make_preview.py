"""Make a labelled contact sheet from actual exported observations."""
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont
from nisar_water.common import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.root
    canvas = Image.new("RGB", (1220, 1140), "#101923")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    regular = ImageFont.truetype(str(font_path), 20)
    title_font = ImageFont.truetype(str(font_path), 30)
    small = ImageFont.truetype(str(font_path), 17)
    draw.text((30, 15), "NISAR water extent | experimental spatial-context model", font=title_font, fill="white")
    draw.text((30, 59), "Blue: predicted water    Cyan: raster-derived boundary    Black: outside observation", font=regular, fill="#acbed0")
    cases = [("504e53bdc8433438", "Oroville reservoir"), ("7fe7a0da7165f375", "Iowa agricultural landscape")]
    for row, (key, name) in enumerate(cases):
        series_dir = root / key
        s = read_json(series_dir / "series.json")
        top = 105 + row*470
        draw.text((30, top), name, font=regular, fill="#76d5ff")
        for col, result in enumerate(s["observations"]):
            left = 30+col*600
            draw.text((left, top+32), result["datetime"][:10], font=regular, fill="white")
            path = series_dir / result["assets"]["composite"]["url"].removeprefix("/assets/")
            with Image.open(path) as raw:
                raw = raw.convert("RGBA")
                raw.thumbnail((565, 375))
                backing = Image.new("RGBA", raw.size, (0, 0, 0, 255))
                image = Image.alpha_composite(backing, raw).convert("RGB")
            canvas.paste(image, (left+(565-image.width)//2, top+64))
            draw.text((left, top+441), f"Mapped water: {result['statistics']['waterHa']:.1f} ha", font=small, fill="#c0d2e1")
    draw.text((30, 1060), "Validation against existing weak labels: precision 81.0%, recall 58.1%. Joint acceptance FAILED.", font=regular, fill="#ffd18a")
    draw.text((30, 1092), "These are model estimates on real radar observations; no flood cause or human impact is established.", font=small, fill="#acbed0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

