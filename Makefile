# G.R.A.P.H AI — top-level orchestration

PY = .venv/Scripts/python.exe
PIP = .venv/Scripts/pip.exe
UVICORN = .venv/Scripts/uvicorn.exe

.PHONY: help install install-frontend dev backend frontend test clean

help:
	@echo "G.R.A.P.H AI Makefile"
	@echo ""
	@echo "  make install          install backend Python deps"
	@echo "  make install-frontend npm install in frontend/"
	@echo "  make backend          run FastAPI at :8000"
	@echo "  make frontend         run Next.js at :3000"
	@echo "  make dev              run backend AND frontend in parallel"
	@echo "  make test             run pytest"
	@echo ""

install:
	$(PIP) install -r requirements.txt

install-frontend:
	cd frontend && npm install

backend:
	PYTHONPATH=. PYTHONIOENCODING=utf-8 $(UVICORN) src.api.main:app --port 8000 --reload

frontend:
	cd frontend && npm run dev

# `make dev` brings up both services. On Windows we spawn two background
# processes; on macOS/Linux this also works via `&`.
dev:
	@echo "Starting backend on :8000 and frontend on :3000 ..."
	@(PYTHONPATH=. PYTHONIOENCODING=utf-8 $(UVICORN) src.api.main:app --port 8000 --reload &) ; \
	  (cd frontend && npm run dev)

test:
	PYTHONPATH=. PYTHONIOENCODING=utf-8 $(PY) -m pytest tests/

clean:
	rm -rf frontend/.next frontend/node_modules logs/alerts.db
