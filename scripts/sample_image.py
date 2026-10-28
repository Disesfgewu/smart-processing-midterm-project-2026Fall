"""Fetch-on-demand helper for the shared feasibility sample image.

tests/data/sample_cats.jpg is intentionally NOT committed: its individual
Flickr source photo's license was never independently verified (see #25
review), so it is not redistributed from this repository. Callers get the
same COCO val2017 000000039769.jpg on demand instead.
"""
import urllib.request
from pathlib import Path

SAMPLE_IMAGE_PATH = "tests/data/sample_cats.jpg"
# http, not https: images.cocodataset.org's TLS certificate is mismatched
# (serves an s3.amazonaws.com cert, SNI mismatch) as of this writing --
# verified with `curl -v https://images.cocodataset.org/...`, which fails
# with "SSL: no alternative certificate subject name matches target host
# name". Using https here would break the download, not harden it.
SAMPLE_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
DOWNLOAD_TIMEOUT_SECONDS = 30


def ensure_sample_image(path: str = SAMPLE_IMAGE_PATH) -> str:
    dest = Path(path)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SAMPLE_IMAGE_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            dest.write_bytes(response.read())
    return str(dest)
