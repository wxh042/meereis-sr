import argparse
import sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "platform"))
from celery_worker import run_super_resolution
p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", default="output.png")
p.add_argument("--model", choices=["10km", "25km"], default="10km")
a = p.parse_args()
image = Image.open(a.input)
gray, _, _ = run_super_resolution(image, a.model)
gray.save(a.output)
print(f"{image.size} -> {gray.size}: {a.output}")
