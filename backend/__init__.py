"""FairFace Guard backend."""

import os as _os

# OpenCV reads this once, when it is first imported. It makes the decoder
# refuse an image by its declared dimensions instead of allocating the pixels
# first, which is what stops a few hundred kilobytes of crafted PNG from
# costing gigabytes of memory. Keep in step with config.load_settings().
# This is a second line of defence only: it works when this package is
# imported before cv2 and is easy to defeat by accident, so the pipeline also
# reads the declared dimensions from the file header itself before decoding
# (backend.engine.pipeline.declared_size).
try:
    _limit = int(_os.environ.get("FAIRFACE_MAX_FRAME_PIXELS") or 4096 * 4096)
    _existing = int(_os.environ.get("OPENCV_IO_MAX_IMAGE_PIXELS") or _limit)
    _os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(min(_existing, _limit))
except ValueError:
    _os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(4096 * 4096)
