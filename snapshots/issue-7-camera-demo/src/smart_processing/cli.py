"""Thin CLI entry point. Detection/segmentation commands land in later issues."""
import argparse

from smart_processing import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="smart-processing",
        description="Text-prompted open-vocabulary detection and segmentation pipeline.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args(argv)
    print(
        "smart-processing is installed (package/environment foundation only). "
        "Detection and segmentation commands are not implemented yet; see #3-#8."
    )


if __name__ == "__main__":
    main()
