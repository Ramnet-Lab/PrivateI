FROM python:3.12-slim

# poppler-utils renders PDF pages; tesseract reads printed text and decides
# which pages need the vision model; libglib2.0-0 is the opencv runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# vis-network is vendored at build time so the graph page needs no network when
# it is opened.  pyvis is installed only to source the file, then removed.
RUN pip install --no-cache-dir pyvis==0.3.2 \
    && python -c "\
import pathlib, shutil, pyvis;\
root = pathlib.Path(pyvis.__file__).parent;\
matches = sorted(root.rglob('vis-network.min.js'));\
assert matches, 'vis-network.min.js not found in the pyvis package';\
target = pathlib.Path('/app/static');\
target.mkdir(parents=True, exist_ok=True);\
shutil.copy(matches[0], target / 'vis-network.min.js');\
print('vendored', matches[0])" \
    && pip uninstall -y pyvis >/dev/null

COPY app/pipeline /app/pipeline
COPY app/templates /app/templates
COPY app/static /app/static
COPY app/main.py /app/main.py

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
