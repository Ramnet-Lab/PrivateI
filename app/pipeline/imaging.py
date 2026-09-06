"""Page image normalisation.

A scan and a photograph are not the same problem, so they do not get the same
treatment.

A scan arrives flat, lit evenly and square to the sensor.  All it needs is
deskew and contrast, and that is all it gets: no denoise that eats faint
pencil, no binarisation that drops a light annotation.

A photograph arrives with the page somewhere inside a larger frame, tilted in
three dimensions rather than two, lit from one side, and rotated ninety degrees
because the phone recorded the rotation in EXIF instead of applying it.  Deskew
cannot help with any of that - it estimates one angle over the whole frame,
which on a photograph means it measures the desk as much as the page.  So the
photograph path first finds the sheet and warps it flat, then divides out the
lighting, and only then runs the scan path over what is left.

The red lines are unchanged.  Nothing here binarises, and nothing here
denoises: the marks that survive are the marks that were on the page.  The
source file in 01_raw remains the authority; these images exist to make OCR and
the vision model work, and any doubt is resolved by looking at the original.

The last function is different in kind.  fit_for_model() does not produce a
page image at all - it produces the copy that goes into an HTTP request, sized
to what a model endpoint will accept.  It is here rather than in the model
client because it is an imaging decision, and it is separate from normalize()
because what a model should see and what an operator should be shown are two
different things.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import env_int

MAX_SKEW_DEG = 15.0     # beyond this the estimate is noise, not skew
MIN_DIM = 400           # do not try to deskew a thumbnail

# --- finding the sheet inside a photograph ---------------------------------
#
# Every bound here exists to make a wrong answer impossible rather than to make
# a right one likely.  A missed page costs a crop that would have helped; a
# wrong page costs evidence, because the warp would cut content away and the
# transcript would be of what was left. So each test below is a reason to
# refuse, and refusing means the full frame goes through unchanged.
DETECT_LONG_EDGE = 1000     # find the quad on a small copy, warp the big one
PAGE_MIN_AREA = 0.25        # a quad this small is a label or a photo on a desk
# A quad this large is the frame's own border. Finding "a page" that is the
# whole picture is not a finding: the warp would be an identity transform, and
# saying page_found about it turns the one useful line in the log into noise.
PAGE_MAX_AREA = 0.97
MASK_MAX_FOREGROUND = 0.98  # a mask this full segmented nothing; ignore it
PAGE_MIN_ASPECT = 0.35      # no paper is this long and thin
PAGE_MAX_ASPECT = 2.9
PAGE_MIN_SIDE_RATIO = 0.6   # opposite sides of a page are roughly equal
PAGE_MAX_CORNER_SKEW = 40.0  # degrees a corner may sit away from square

# --- dividing out the lighting ---------------------------------------------
FLATTEN_LONG_EDGE = 800     # the lighting is low-frequency; estimate it small
FLATTEN_MAX_GAIN = 2.5      # a cap, so a genuinely dark region stays dark

# --- contrast and sharpening ------------------------------------------------
PHOTO_CLAHE_CLIP = 1.2      # see enhance(): a flattened page needs far less
PHOTO_CLAHE_TILES = 4
SHARPEN_AMOUNT = 0.6        # mild on purpose: ringing invents strokes
SHARPEN_RADIUS = 1.6

# --- the copy that goes to a model ------------------------------------------
#
# Docker Model Runner refuses a request body over 10 MiB - measured exactly,
# byte for byte, against the running endpoint - and answers "request too large"
# before it looks at anything.  Base64 costs a third on top of the file, so the
# ceiling on the image itself is about 7.5 MB.  MODEL_MAX_BYTES sits well under
# that, and MODEL_MAX_PX sits far under MODEL_MAX_BYTES, because the limit is
# not the reason to resize.
#
# The reason to resize is that the pixels are not used. Measured against this
# deployment's own model: the same page at 1024, 1568, 2048 and 4000 pixels
# costs 512 prompt tokens every time, and comes back with the same reading.
# The vision encoder works to a fixed grid, so everything above it is resampled
# away after being uploaded. It is a default rather than a constant because
# that is a fact about one model - an endpoint that tiles high-resolution input
# would use more - so VLM_IMAGE_MAX_PX raises it without a code change.
MODEL_MAX_PX = env_int("VLM_IMAGE_MAX_PX", 2048)
MODEL_JPEG_QUALITY = 90
MODEL_MAX_BYTES = 7_000_000
MODEL_MIN_PX = 640          # below this a page is not worth sending at all


def _estimate_skew(gray: np.ndarray) -> float:
    """Angle in degrees from the minimum-area rectangle around dark pixels."""
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thresh)
    if coords is None or len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    return float(angle)


def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if min(h, w) < MIN_DIM:
        return image, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    angle = _estimate_skew(gray)
    if abs(angle) < 0.2 or abs(angle) > MAX_SKEW_DEG:
        return image, 0.0
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def enhance(image: np.ndarray, *, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """CLAHE on the luminance channel.

    Local contrast, so faint pencil on one part of a page lifts without
    blowing out a dark photocopied block elsewhere.  Colour is preserved -
    ink colour sometimes carries meaning in an investigative note.

    A photograph asks for less of this than a scan, which is why the strength
    is a parameter. On a page whose lighting has already been divided out, the
    scan's setting has little real unevenness left to correct and spends itself
    on what is left: paper grain, and the printing showing through from the
    reverse of the sheet. Measured on the page that prompted this work, the
    scan setting raised the show-through visibly and the strokes not at all -
    it was manufacturing marks that are not on this side of the paper. So the
    photograph path asks for a gentler lift over larger tiles, which keeps the
    help for faint pencil without the invention.
    """
    grid = (tiles, tiles)
    if image.ndim == 2:
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------
# the photograph path
# --------------------------------------------------------------------------
def _order_quad(points: np.ndarray) -> np.ndarray:
    """Four corners as top-left, top-right, bottom-right, bottom-left.

    By sums and differences rather than by angle, because the sums and
    differences of a quadrilateral's coordinates identify its corners whatever
    order findContours walked them in and whichever way up the page is.
    """
    pts = points.reshape(4, 2).astype(np.float32)
    total = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(total)], pts[np.argmin(diff)],
                     pts[np.argmax(total)], pts[np.argmax(diff)]],
                    dtype=np.float32)


def _side_lengths(quad: np.ndarray) -> tuple[float, float, float, float]:
    tl, tr, br, bl = quad
    return (float(np.linalg.norm(tr - tl)), float(np.linalg.norm(br - bl)),
            float(np.linalg.norm(bl - tl)), float(np.linalg.norm(br - tr)))


def _is_plausible_page(quad: np.ndarray, frame_area: float) -> bool:
    """Whether a quadrilateral is shaped like a sheet of paper seen at an angle."""
    area = float(cv2.contourArea(quad))
    if not frame_area or not PAGE_MIN_AREA <= area / frame_area <= PAGE_MAX_AREA:
        return False

    top, bottom, left, right = _side_lengths(quad)
    if min(top, bottom, left, right) <= 0:
        return False
    # Perspective shortens the far edge, but only so far. A quad whose opposite
    # sides differ by more than this is a shadow or a desk edge caught in the
    # threshold, not a rectangle photographed off-axis.
    if (min(top, bottom) / max(top, bottom) < PAGE_MIN_SIDE_RATIO
            or min(left, right) / max(left, right) < PAGE_MIN_SIDE_RATIO):
        return False

    aspect = max(top, bottom) / max(left, right)
    if not PAGE_MIN_ASPECT <= aspect <= PAGE_MAX_ASPECT:
        return False

    for i in range(4):
        a = quad[(i - 1) % 4] - quad[i]
        b = quad[(i + 1) % 4] - quad[i]
        norms = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        if norms <= 0:
            return False
        angle = np.degrees(np.arccos(np.clip(float(np.dot(a, b)) / norms, -1.0, 1.0)))
        if abs(angle - 90.0) > PAGE_MAX_CORNER_SKEW:
            return False
    return True


def _page_masks(gray: np.ndarray) -> list[np.ndarray]:
    """Two ways of seeing the sheet, because one of them is often wrong.

    Brightness finds paper on a dark desk and misses paper on a light one.
    Edges find the boundary whatever the two sides are worth, and lose it in a
    busy background.  Both are tried and the best-scoring quad wins, so a
    background that defeats one is answered by the other.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bright = cv2.threshold(blurred, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    edges = cv2.dilate(cv2.Canny(blurred, 50, 150),
                       np.ones((5, 5), np.uint8), iterations=1)
    return [bright, edges]


def find_page(image: np.ndarray) -> np.ndarray | None:
    """The page's four corners in image coordinates, or None if unsure.

    None is the safe answer and is returned freely.  The caller keeps the whole
    frame when this finds nothing, which costs a little of the model's
    attention; a wrong quad would cost part of the page.
    """
    h, w = image.shape[:2]
    if min(h, w) < MIN_DIM:
        return None

    scale = min(1.0, DETECT_LONG_EDGE / max(h, w))
    small = (cv2.resize(image, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    frame_area = float(gray.shape[0] * gray.shape[1])

    best: np.ndarray | None = None
    best_area = 0.0
    for mask in _page_masks(gray):
        if float((mask > 0).mean()) > MASK_MAX_FOREGROUND:
            # Everything is foreground, so nothing was separated from anything.
            # Left in, this contributes the frame's own rectangle as a
            # candidate, which passes every shape test there is.
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(contour) / frame_area < PAGE_MIN_AREA:
                break                     # sorted, so every one after is smaller
            approx = cv2.approxPolyDP(
                contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = _order_quad(approx)
            if not _is_plausible_page(quad, frame_area):
                continue
            area = float(cv2.contourArea(quad))
            if area > best_area:
                best, best_area = quad, area

    return None if best is None else best / scale


def warp_page(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Flatten the sheet onto a rectangle the size of its longest edges.

    Sized by the longest opposite pair rather than an average, so the near edge
    keeps its resolution and the far one is stretched up to meet it. The other
    way round would throw away pixels on the part of the page the camera saw
    best.
    """
    top, bottom, left, right = _side_lengths(quad)
    width = int(round(max(top, bottom)))
    height = int(round(max(left, right)))
    if width < MIN_DIM or height < MIN_DIM:
        return image
    target = np.array([[0, 0], [width - 1, 0],
                       [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, target)
    return cv2.warpPerspective(image, matrix, (width, height),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def flatten_illumination(image: np.ndarray) -> np.ndarray:
    """Divide out the lighting and keep the marks.

    A phone photograph of a page is brighter on the lamp side and shades off
    towards a shadow, and a single global contrast curve cannot serve both ends
    of that gradient - lift the dark end and the bright end clips, protect the
    bright end and the dark end stays muddy.

    So the lighting is estimated and removed rather than compensated for.
    Closing the image with a kernel far wider than any pen stroke erases the
    writing and leaves the paper, which is a picture of the light that fell on
    it; dividing by that levels the paper and leaves every stroke where it was.
    Strokes keep their contrast against their own neighbourhood, which is what
    makes this different from brightening: nothing faint is pushed towards the
    paper's value.

    The estimate is made on a small copy because lighting has no fine detail,
    and the gain is capped because a genuinely dark region - a photocopied
    block, an inked stamp - must not be scrubbed up into paper.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]

    scale = min(1.0, FLATTEN_LONG_EDGE / max(h, w))
    small = (cv2.resize(gray, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else gray)
    span = max(15, (min(small.shape[:2]) // 8) | 1)
    paper = cv2.morphologyEx(
        small, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span)))
    paper = cv2.GaussianBlur(paper, (0, 0), span / 6.0)
    paper = cv2.resize(paper, (w, h), interpolation=cv2.INTER_LINEAR)

    # The brightest paper is the reference: everywhere else is lifted to match
    # it, rather than everything being lifted towards white, which would clip.
    target = float(np.percentile(paper, 90))
    gain = np.clip(target / np.maximum(paper.astype(np.float32), 1.0),
                   1.0 / FLATTEN_MAX_GAIN, FLATTEN_MAX_GAIN)
    if image.ndim == 3:
        gain = gain[..., None]
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def sharpen(image: np.ndarray) -> np.ndarray:
    """Unsharp mask, deliberately weak.

    A camera's own blur costs a stroke its edge, and an edge is most of what a
    reader - human or model - has to go on. Strong sharpening buys that back
    with ringing, and a ring around a stroke is a mark that was not on the
    page, which on this pipeline's terms is worse than a soft one that was.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), SHARPEN_RADIUS)
    return cv2.addWeighted(image, 1.0 + SHARPEN_AMOUNT, blurred,
                           -SHARPEN_AMOUNT, 0)


def normalize(image: np.ndarray, *, photo: bool = False) -> tuple[np.ndarray, dict]:
    """The page as OCR and the model should see it, and what was done to it.

    photo=True adds the three steps a camera makes necessary and a scanner does
    not. It is the caller's statement about where the pixels came from, not a
    guess made here: a rendered PDF page is never a photograph however it
    looks, and running page-finding over one could only ever lose a margin.
    """
    meta: dict = {"photo": photo, "page_found": None}
    if photo:
        quad = find_page(image)
        meta["page_found"] = quad is not None
        if quad is not None:
            before = image.shape[:2]
            image = warp_page(image, quad)
            meta["cropped_from"] = f"{before[1]}x{before[0]}"
        image = flatten_illumination(image)

    out, angle = deskew(image)
    if photo:
        out = sharpen(enhance(out, clip=PHOTO_CLAHE_CLIP, tiles=PHOTO_CLAHE_TILES))
    else:
        out = enhance(out)

    h, w = out.shape[:2]
    meta.update({"deskew_deg": round(angle, 3), "width": int(w), "height": int(h)})
    return out, meta


# --------------------------------------------------------------------------
# the copy a model is sent
# --------------------------------------------------------------------------
def fit_for_model(source: Path, dest: Path, *, max_px: int = MODEL_MAX_PX,
                  quality: int = MODEL_JPEG_QUALITY,
                  max_bytes: int = MODEL_MAX_BYTES) -> tuple[Path, dict]:
    """Write a copy of a page image that a model endpoint will accept.

    JPEG, not PNG. A page image is a photograph of a document by the time it
    reaches here, and PNG stores photographs about twenty times larger than
    JPEG does at a quality no reader can tell apart - on the page that prompted
    this, 17.8 MB against 0.9 MB. The whole of that difference is upload time
    on every single request.

    Two limits, and they are limits of different kinds. max_px is a judgement
    about usefulness: a vision model tiles its input down to a couple of
    megapixels whatever it is handed, so pixels above that are paid for and
    discarded. max_bytes is a hard fact about the endpoint, which refuses a
    body over 10 MiB outright, so the loop below keeps shrinking until the file
    is under it rather than trusting the first estimate.

    Returns the path written and what it did, so the caller can log the size a
    request actually carried instead of the size on disk.
    """
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read the page image at {source}")

    h, w = image.shape[:2]
    scale = min(1.0, max_px / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(round(w * scale))),
                                   max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)

    dest.parent.mkdir(parents=True, exist_ok=True)
    encoded: np.ndarray | None = None
    for attempt in range(6):
        ok, buffer = cv2.imencode(".jpg", image,
                                  [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError(f"could not encode {source.name} as JPEG")
        encoded = buffer
        if buffer.nbytes <= max_bytes:
            break
        # Still too big for the endpoint. Halve the pixels rather than drop the
        # quality further: at this size the detail is gone either way, and a
        # smaller sharp page reads better than a large mushy one.
        side = max(image.shape[:2])
        if side <= MODEL_MIN_PX:
            break
        image = cv2.resize(image, None, fx=0.7, fy=0.7,
                           interpolation=cv2.INTER_AREA)

    # A .partial name first, as the page images do: a reader that arrives
    # mid-write must not find a truncated JPEG under the real name.
    tmp = dest.with_name(f"{dest.stem}.partial.jpg")
    tmp.write_bytes(encoded.tobytes())
    tmp.replace(dest)
    return dest, {"width": int(image.shape[1]), "height": int(image.shape[0]),
                  "bytes": int(encoded.nbytes),
                  "source_bytes": int(source.stat().st_size)}
