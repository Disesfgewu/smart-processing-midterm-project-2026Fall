"""Fetch-on-demand helper for the shared feasibility sample image.

tests/data/sample_cats.jpg is intentionally NOT committed: its individual
Flickr source photo's license was never independently verified (see #25
review), so it is not redistributed from this repository. Callers get the
same COCO val2017 000000039769.jpg on demand instead.
"""
import urllib.request
from pathlib import Path

SAMPLE_IMAGE_PATH = "tests/data/sample_cats.jpg"
SAMPLE_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def ensure_sample_image(path: str = SAMPLE_IMAGE_PATH) -> str:
    dest = Path(path)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAMPLE_IMAGE_URL, dest)
    return str(dest)
