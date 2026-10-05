up:        ; docker compose up --build
test:      ; LLM_PROVIDER=none pytest -q
eval:      ; LLM_PROVIDER=none python -m evals.run_evals
sample:    ; python scripts/make_sample_data.py
dev:       ; uvicorn app.main:app --reload
