"""Lay a Word file out into pages with LibreOffice, and check that it did.

A .docx has no pages.  Word computes them at display time, and so does every
citation this system produces, which is why a Word file is rendered rather than
dumped: the rendering is the only thing that assigns a page number a finding can
point at, and the only thing that computes list numbering and field values the
file does not store.

The renderer is treated as unreliable on purpose, because it is.  Its exit code
does not report whether a document was produced - measured, 3 of 6 parallel jobs
exited 1, wrote no file at all, and printed nothing but the cosmetic javaldx
warning; the reverse has also been measured, exit 0 with nothing on disk.  So
nothing here believes soffice about its own success.  The file on disk decides,
it is opened with the same reader the ingest will use, and it must report at
least one page or the run counts as a failure.

Failure is a result, not an exception: a Word file that cannot be laid out is
still readable from its own XML, and the caller needs the reason in order to say
so on the document page.  Only a caller mistake - a source file that is not
there - raises.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .config import env_int, env_str
from .log import get_logger

log = get_logger("docx_convert")


class ConversionFailed(RuntimeError):
    pass


@dataclass
class Conversion:
    """What the renderer did, in the terms the caller has to record it in."""
    pdf: Path | None
    pages: int
    message: str


# ExportNotes carries Word comments into the PDF.  Without it the export drops
# them outright - measured twice, pypdf reports zero annotations.  With it they
# arrive as /Text annotations, which no text extractor returns, so the comments
# are read off the annotations themselves further down the pipeline.
# ExportNotesInMargin is NOT a valid property on LibreOffice 25.2 and fails the
# whole export with SfxBaseModel::impl_store failed: 0xc10, writing no file at
# all - do not add it.
_FILTER = ('pdf:writer_pdf_Export:'
           '{"ExportNotes":{"type":"boolean","value":"true"}}')


def _kill_group(pid: int | None) -> None:
    # start_new_session made the child a session leader, so its process group id
    # is its pid and killing the group takes soffice.bin with the launcher that
    # forked it.  Asking getpgid() first would be worse: subprocess.run has
    # already killed and reaped the direct child by the time TimeoutExpired
    # reaches us, so the lookup raises while the group is still alive.
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass                        # already gone; nothing left to kill


def _run_once(soffice: str, target: str, src: Path, dest: Path,
              timeout: int) -> tuple[Path | None, str]:
    """One soffice run.  Returns the file it actually wrote, and its last word."""
    # A private profile per run.  Two soffice processes sharing one profile lose
    # documents: measured, 3 of 6 parallel jobs exited 1 and wrote nothing.
    # Ingestion is already serialised on the single worker thread, so this is
    # the second lock rather than the first - its real job is that a container
    # killed mid-conversion leaves a .~lock in a shared profile that wedges
    # every later run, and a profile created and destroyed per run cannot.  The
    # profile is passed explicitly because soffice finds it through getpwuid(),
    # not $HOME, and dies with 'User installation could not be completed' the
    # day a USER line is added to the Dockerfile.
    #
    # The output directory sits beside the destination, not in /tmp, because the
    # finished file is moved with os.replace and /tmp is a different filesystem
    # from the data volume: staging the PDF in /tmp would convert the document
    # successfully and then lose it to EXDEV.
    with tempfile.TemporaryDirectory(prefix="soffice_") as home, \
            tempfile.TemporaryDirectory(prefix="convert_", dir=dest.parent) as out:
        cmd = [soffice, "--headless", "--norestore", "--nolockcheck",
               "--nodefault", "--nofirststartwizard",
               # A directory that does not exist yet: soffice builds the profile
               # itself, and handing it one already there has been the
               # difference between a clean start and a half-made profile.
               f"-env:UserInstallation=file://{home}/profile",
               "--convert-to", target, "--outdir", out, str(src)]
        try:
            # Popen rather than subprocess.run, for one reason: run() raises a
            # TimeoutExpired that carries no pid, so there is nothing to kill
            # the process group by, and a hung soffice would outlive the ingest
            # that started it.  Owning the pid is the whole point.
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True,
                                    start_new_session=True)
        except OSError as exc:      # the binary went away after which() found it
            return None, f"LibreOffice could not be started ({exc})"
        try:
            spoke, complained = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc.pid)
            proc.communicate()      # reap it, so no zombie outlives the ingest
            return None, f"LibreOffice gave no answer in {timeout}s"
        said = (spoke + complained).strip().splitlines()
        last = said[-1].strip() if said else f"exit {proc.returncode}"

        produced = sorted(Path(out).glob("*.pdf"))
        if not produced:
            return None, last
        # Out of the temporary directory before it is swept, into the document's
        # own page directory under a partial name: a half-written render must
        # never be mistaken for the finished one if the container dies here.
        staged = dest.with_name(f"{dest.stem}.partial.pdf")
        os.replace(produced[0], staged)
        return staged, last


def _page_count(pdf: Path) -> int:
    from pypdf import PdfReader
    return len(PdfReader(str(pdf)).pages)


def render(src: Path, dest: Path, *, timeout: int | None = None) -> Conversion:
    """Render src to dest, reporting what happened rather than raising."""
    if not src.is_file():
        raise FileNotFoundError(f"there is no file to convert at {src}")

    soffice = env_str("SOFFICE_BIN", "soffice")
    if not shutil.which(soffice):
        return Conversion(None, 0, f"{soffice} is not installed in this image")
    if timeout is None:
        timeout = env_int("DOCX_CONVERT_TIMEOUT", 300)
    dest.parent.mkdir(parents=True, exist_ok=True)

    message = ""
    # The filtered run first, then the same run without the filter.  A filter
    # string rejected by a future LibreOffice would otherwise drop the whole
    # document to the fallback path and lose real page numbers over a comment
    # setting; the comments then reach the corpus through the annex instead.
    for target in (_FILTER, "pdf"):
        notes = target is _FILTER
        made, said = _run_once(soffice, target, src, dest, timeout)
        if made is None:
            message = said
            log.warning("%s: the run %s the notes filter wrote no file (%s)",
                        src.name, "with" if notes else "without", said)
            continue
        # The file on disk decides, never the return code.  Opening it here also
        # means a truncated or empty render is caught now, by the same reader
        # the ingest will use, rather than three stages later.
        try:
            pages = _page_count(made)
            if pages == 0:
                raise ValueError("it reports zero pages")
        except Exception as exc:
            made.unlink(missing_ok=True)
            message = f"the PDF LibreOffice wrote could not be read ({exc})"
            log.warning("%s: %s", src.name, message)
            continue
        os.replace(made, dest)
        return Conversion(dest, pages, "" if notes else
                          f"exported without the notes filter, so Word comments "
                          f"are not in the rendering ({said})")
    return Conversion(None, 0, message or "LibreOffice produced no usable PDF")


def convert(src: Path, doc_id: str) -> Path:
    """Render src into the document's own directory; raise if nothing arrived."""
    result = render(src, paths.converted_pdf(doc_id))
    if result.pdf is None:
        raise ConversionFailed(result.message)
    if result.message:
        log.warning("%s: %s", doc_id, result.message)
    log.info("%s: laid out as %d page(s)", doc_id, result.pages)
    return result.pdf
