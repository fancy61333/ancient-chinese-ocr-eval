#!/usr/bin/env python3
"""Validate a prediction file against the submission schema.

The scorer accepts, per image, a list of records carrying a bounding box and a
predicted character. This utility checks that shape before submission: keys are
image identifiers with no file extension, values are lists, each record has the
required fields with the right types, and boxes are non-degenerate.

It does not check correctness against ground truth — that is done by the platform
scorer, which is not public.

Usage:
    python validate_prediction_format.py prediction.json
"""

import json
import sys
from pathlib import Path

REQUIRED_FIELDS = ("bbox", "text")


def validate_prediction_json(json_path: Path) -> bool:
    if not json_path.exists():
        print(f"FAIL file not found: {json_path}")
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"FAIL not valid JSON: {e}")
        return False

    if not isinstance(data, dict):
        print("FAIL root node is not an object")
        return False

    ok = True
    total_images = 0
    total_chars = 0
    issues = []

    for image_id, predictions in data.items():
        total_images += 1

        if "." in image_id:
            issues.append(f"image id contains a file extension: {image_id}")
            ok = False

        if not isinstance(predictions, list):
            issues.append(f"predictions for {image_id} are not a list")
            ok = False
            continue

        for rec in predictions:
            total_chars += 1

            if not isinstance(rec, dict):
                issues.append(f"{image_id}: record is not an object")
                ok = False
                continue

            for field in REQUIRED_FIELDS:
                if field not in rec:
                    issues.append(f"{image_id}: missing field '{field}'")
                    ok = False

            bbox = rec.get("bbox")
            if isinstance(bbox, list):
                if len(bbox) != 4:
                    issues.append(f"{image_id}: bbox has {len(bbox)} values, expected 4")
                    ok = False
                elif not all(isinstance(v, (int, float)) for v in bbox):
                    issues.append(f"{image_id}: bbox values are not numeric")
                    ok = False
                elif bbox[2] <= 0 or bbox[3] <= 0:
                    issues.append(f"{image_id}: bbox has non-positive width or height")
                    ok = False
            elif bbox is not None:
                issues.append(f"{image_id}: bbox is not a list")
                ok = False

            if "text" in rec and not isinstance(rec["text"], str):
                issues.append(f"{image_id}: text is not a string")
                ok = False

    print(f"images: {total_images}   predicted characters: {total_chars}")

    if issues:
        print(f"{len(issues)} issue(s); first 20 shown:")
        for line in issues[:20]:
            print("  -", line)
    else:
        print("no structural issues found")

    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(0 if validate_prediction_json(Path(sys.argv[1])) else 1)
