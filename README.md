HydroFlow
=========

Quickstart
----------

1. Create environment and install:

   - python3 -m venv .venv && source .venv/bin/activate
   - pip install -e .
   - pip install -e .[dev]

2. Run tests:

   - pytest -q

3. CLI:

   - hydroflow info --check
   - hydroflow analyze bathymetry data.xyz --method idw --resolution 2.0
   - hydroflow analyze sediment flow.csv --d50 0.5 --d90 1.0

4. API:

   - hydroflow api serve --host 0.0.0.0 --port 8000

5. OpenAI Agent (optional):

   - export OPENAI_API_KEY=...
   - hydroflow agents summarize --metrics metrics.json


