# PrivateI

Upload documents, they get read and processed, and the people, places, and
events in them appear as a graph you can click through.

Everything runs on this Mac. After the models are pulled, nothing leaves it.

## What it does

1. You drop in PDFs, Word files, or images.
2. Each page is read — from the PDF's own text layer where there is one, by OCR
   where the page is clean printed text, and by a vision model where it is
   handwriting or a poor scan.
3. A text model turns each page into statements: who did what, where, when.
4. Those statements become a graph, with the exact quote and page behind every
   connection.

## Setup

```bash
./start.sh
```

That is the whole thing. It checks for Docker (installing Docker Desktop if you
do not have it), turns on Docker Model Runner, writes a `.env` with a random
database password, builds the images, starts everything, downloads the models
(~8 GB, once), proves the model answers, and opens your browser.

Safe to run again — it never overwrites your `.env` and never re-downloads a
model you already have.

### Windows

```
start-windows.cmd
```

Double-click it (or run `.\start.ps1` in PowerShell). Same flow as the Mac
script: Docker Desktop via winget if missing, Model Runner enabled, models
pulled, an inference proof, then the browser. On a machine with an AMD or
Intel GPU, Model Runner uses its Vulkan backend automatically - no drivers to
configure beyond the card's own. The script surfaces `docker model status` so
you can see which backend answered; if it shows no GPU backend, update Docker
Desktop (Vulkan support shipped in late 2025 and early builds had upgrade
regressions).

Plain-speak hardware note: the model is ~7.3 GB of weights. A card with 8 GB+
of VRAM holds all of it and is fast; less VRAM means llama.cpp splits layers
with system RAM and slows down accordingly. A machine short on both is the
same wall the Mac hit.

The background auto-updater has a Windows twin too: `.\auto-update.ps1`
with `Once`, `Watch`, `Start`, `Stop`, `Status`.

### How the models run

Models execute through **Docker Model Runner** — a Docker Desktop feature that
runs them natively on this machine, which on a Mac means the Metal GPU, managed
entirely by Docker. Nothing model-related runs inside a container.

Measured here with the same 12B-class model:

| Where the model ran | tokens/sec |
|---|---|
| Ollama in a container (no GPU in the Docker VM) | 4.4 |
| Ollama natively on the host | 9.4 |
| **Docker Model Runner (Metal via llama.cpp)** | **35.4** |

Because the weights never enter the Docker VM, the VM needs memory only for the
app and Neo4j — there is no VM memory requirement for the models.

These are reasoning models: they think before they answer, and the thinking is
returned separately (you never see it in results, but it is why answers are not
instant). Reply lengths are uncapped by default and the request timeout is the
backstop; see `.env.example`.

### Choosing different models

Models are Docker Hub references. Browse `hub.docker.com/u/ai`, then:

```bash
docker model pull ai/<something>
```

and put the name in `.env`. Changing `EMBED_MODEL` changes the vector width, so
re-index afterwards from the Chat page ("Index passages") — the app skips
vectors whose width does not match rather than mixing them.

## The pages

- **Documents** — drag and drop, and watch each file's progress. Open one to see
  every page image beside the text read from it, plus the facts extracted.
- **Chat** — ask questions about everything you have uploaded.
- **Graph** — the whole picture. Click a node for everything it connects to, each
  with the sentence it came from and a link to the source page. Filter by name or
  by type.
- **Timeline** — every dated statement in order, with its quote and source.

## Chat

Two kinds of context go to the model, because they answer different questions:

- **Passages** — the actual wording of your pages, found by embedding similarity.
  These answer "what did the memo say about X".
- **Relationships** — the graph's edges around whatever you asked about. These
  answer "who else was at Building 220", where no single passage contains the
  answer because it is spread across documents.

The model is told to answer only from that material and to cite `[file p.N]`, and
every answer shows the pages it was given, each linking to the page image. If the
documents do not contain the answer it is instructed to say so rather than fill
the gap.

Passages are indexed automatically as documents are processed. If you set
`EMBED_MODEL` after uploading, use the "Index passages" link on the chat page to
catch up. Without an embedding model chat still works, falling back to keyword
matching, which is noticeably worse at finding the right page.

## What gets accepted

PDF, Word `.docx`, and images (`.png .jpg .jpeg .tif .tiff .bmp .heic .webp`).

Legacy `.doc` is not supported — save it as `.docx` first. Uploading the same
file twice is detected by content hash and skipped, so re-dropping a folder is
harmless. Files above `MAX_UPLOAD_MB` (default 500) are rejected.

## What it will not do

A fact only enters the graph if the model quotes text that actually appears on
the page it cites. Quotes that cannot be found are discarded — that check is
what keeps invented statements with plausible-looking citations out of the
graph. The count of discarded items is shown per document.

Names are merged automatically only in clear cases: exact matches once ranks and
punctuation are stripped, an initialism against a full name, a bare surname
against a full name. `SSgt Smith` and `Smith` become one node; two different
people with similar names do not. Raise or lower `MERGE_THRESHOLD` in `.env` if
the balance is wrong for your documents.

## Tuning

| Setting | Does what |
|---|---|
| `OCR_CONFIDENCE_THRESHOLD` | How well OCR must read a page to skip the vision model. Lower it to send fewer pages to the slow path. |
| `OCR_MIN_WORDS` | A page with fewer words than this always goes to the vision model, however confident OCR was. |
| `PDF_DPI` | Rendering resolution for scanned PDFs. Higher reads better and is slower. |
| `MERGE_THRESHOLD` | How similar two names must be to become one entity. |
| `USE_EMBEDDED_TEXT_LAYER` | Read a born-digital PDF's own text instead of rasterising and OCR-ing it. |
| `TEXT_NUM_CTX` | Context window for chat and extraction. Raise it to fit more passages into an answer. |
| `MODEL_THINKING` | Let a reasoning model think before answering. Off by default — see below. |
| `EXTRACT_NUM_PREDICT` / `CHAT_NUM_PREDICT` / `TRANSCRIBE_NUM_PREDICT` | Hard caps on reply length per stage. |

### Reasoning models

Some models spend tokens reasoning before they answer, and that reasoning comes
out of the same budget as the reply. Ollama's default reply length is unlimited,
so a model can think for minutes and return an empty response — which looks
exactly like a hang, because nothing is logged and nothing comes back.

Both halves of that are handled: `MODEL_THINKING=false` is the default, and every
stage caps its reply length. Measured on `gemma4:12b` with an identical prompt
and identical output: 19 tokens in 5.6 s with thinking off, 103 tokens in 25.5 s
with it on. Turn it on only if answer quality on your documents clearly needs it.

Note that the `-mlx` model builds are text-only here: sending one an image
terminates the Ollama server process rather than returning an error.

## Everyday commands

```bash
make up        # start (also rebuilds if code changed)
make logs      # follow what the processor is doing
make down      # stop
make models    # what Ollama has installed
make reset     # delete every document and empty the graph
```

## Where things are kept

```
data/
├── 01_raw/       the files you uploaded, untouched
├── 02_pages/     page images
├── 03_text/      the text read from each page
├── 04_graph_db/  the Neo4j store
└── state.db      what has been processed and what came out
```

One folder. `make reset` empties it; deleting a single document from the UI
removes its pages, its text, and its facts from the graph.

Do not open `data/state.db` from macOS while the app is running. It lives on a
bind mount shared with the Docker VM, and the two sides do not share a coherent
lock view — a container will report `database disk image is malformed` against a
file that is perfectly intact. Stop the app first if you need to look at it.

## Troubleshooting

**Everything says "text only" and mentions the model endpoint.** The app cannot
reach Docker Model Runner. Check `docker model status`, turn it on with
`docker desktop enable model-runner`, then hit Retry on the document.

**A document says "unreadable".** No text could be read from any page, which
means it is a scan needing a vision model. Set `VLM_MODEL` and retry it.

**Neo4j rejects the password after you changed it.** The initial password is
stored on first start. Either change it in the Neo4j browser at
http://127.0.0.1:7474, or `make down`, delete `data/04_graph_db/`, and start again.

**A person appears as two nodes.** Lower `MERGE_THRESHOLD`. If two different
people were merged into one, raise it.
