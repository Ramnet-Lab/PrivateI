SHELL := /bin/bash
COMPOSE := docker compose

.DEFAULT_GOAL := help
.PHONY: help setup up down logs restart rebuild status reset destroy models

help:
	@echo "  ./start.sh     first time? use this - it does everything below"
	@echo "  make setup     create .env and the data folder"
	@echo "  make up        build if needed and start; opens on http://127.0.0.1:8080"
	@echo "  make down      stop everything"
	@echo "  make logs      follow the app log"
	@echo "  make rebuild   rebuild the image after a code change"
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

up: setup
	$(COMPOSE) up -d --build
	@echo
	@echo "  http://127.0.0.1:$${APP_PORT:-8080}"

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f app

restart:
	$(COMPOSE) restart app

rebuild:
	$(COMPOSE) up -d --build app

status:
	@$(COMPOSE) ps

models:
	@docker model list 2>/dev/null || echo "Model Runner is not on - run ./start.sh"

pull:
	@docker model pull $$(grep '^TEXT_MODEL=' .env | cut -d= -f2)
	@docker model pull $$(grep '^EMBED_MODEL=' .env | cut -d= -f2)

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
