SHELL := /bin/bash
COMPOSE := docker compose

.DEFAULT_GOAL := help
.PHONY: help setup up down logs restart rebuild status reset destroy models pull

help:
	@echo "  ./start.sh     first time? use this - it does everything below"
	@echo "  make setup     create .env and the data folder"
	@echo "  make up        build if needed and start; opens on http://127.0.0.1:8080"
	@echo "  make down      stop everything"
	@echo "  make logs      follow the app log"
	@echo "  make rebuild   rebuild the image after a code change"
	@echo "  make pull      download the models named in .env (make up does this)"
	@echo "  note: plain 'docker compose up' skips the model download"
	@echo "  make models    list the models Model Runner has"
	@echo "  make pull      download the models named in .env"
	@echo "  make status    container health"
	@echo "  make reset     empty the graph and delete all uploaded documents"
	@echo "  make destroy   the above, plus remove containers and volumes"

setup:
	@test -f .env || (cp .env.example .env && echo "created .env - set NEO4J_PASSWORD in it")
	@mkdir -p data/01_raw data/02_pages data/03_text data/04_graph_db data/05_products data/99_logs
	@grep -q '^NEO4J_PASSWORD=.\+' .env \
	  || echo "NEO4J_PASSWORD is empty in .env - generate one: openssl rand -base64 24"

up: pull
	$(COMPOSE) up -d --build
	@echo
	@echo "  http://127.0.0.1:$${APP_PORT:-8080}"

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f app

restart:
	$(COMPOSE) restart app

rebuild: pull
	$(COMPOSE) up -d --build app

status:
	@$(COMPOSE) ps

models:
	@docker model list 2>/dev/null || echo "Model Runner is not on - run ./start.sh"

# Models live in the host's Model Runner, never in the image: the container has
# no Docker socket, so nothing at image-build time could fetch them. This is how
# a build gets its models - every target that builds depends on this one. Two
# models, not three: transcription asks for the text model, so there is no
# separate vision model to fetch.
pull: setup
	@./scripts/pull-models.sh

reset:
	@read -r -p "Delete every uploaded document and empty the graph? [y/N] " ok; \
	 if [ "$$ok" = "y" ]; then \
	   $(COMPOSE) down; \
	   rm -rf data/01_raw data/02_pages data/03_text data/04_graph_db data/05_products data/state.db; \
	   mkdir -p data/01_raw data/02_pages data/03_text data/04_graph_db data/05_products; \
	   echo "cleared"; \
	 else echo "nothing deleted"; fi

destroy:
	@read -r -p "Remove all data AND the containers and volumes? [y/N] " ok; \
	 if [ "$$ok" = "y" ]; then \
	   $(COMPOSE) down -v --remove-orphans; rm -rf data; echo "removed"; \
	 else echo "nothing deleted"; fi
