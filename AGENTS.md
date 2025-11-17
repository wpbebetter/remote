# Repository Guidelines

This guide explains how to extend the PEK T3 gate-assignment toolkit, work with its flight datasets, and submit dependable changes.

## Project Structure & Module Organization
- `gate_assignment/` (root package) hosts Python sources: `data/` handles preprocessing, `models/` contains optimizers, and `cli/` exposes runnable entry points.
- `data/raw/` stores immutable vendor CSVs (e.g., `t3cde_candidate_flights_final_v2.csv`, `taxi_distance_matrix_clean.csv`). Never hand-edit these files.
- `data/clean/` holds reproducible outputs produced by the preprocessing module, including `T3CDE_stands_clean.csv` and `PEK_stand_runway_directional_taxi_distances.csv`.
- `reference/` mirrors external papers or baseline implementations. Consult them for context, but do not import or commit modifications.

## Build, Test, and Development Commands
- `conda run -n gate_assignment python -m gate_assignment.data.preprocess`: regenerate every table under `data/clean/` from the raw suppliers and print interpolation statistics; run this before shipping refreshed data.
- `conda run -n gate_assignment python -m pytest`: execute the complete unit/integration suite within `gate_assignment/tests`. Keep tests colocated with the modules they cover.
- `conda run -n gate_assignment python -m gate_assignment.cli.assign --schedule data/clean/t3cde_candidate_flights_final_v2.backup.csv`: run the optimization CLI against the latest cleaned schedule to validate solver changes.

## Coding Style & Naming Conventions
Target Python 3.11, four-space indentation, and `snake_case` identifiers. Public functions must carry type hints and concise docstrings. Keep modules focused on one domain (~400 lines maximum) and name files after the domain entity (`stands.py`, `solver.py`). Always run `black --line-length 100` and `ruff --fix` before committing.

## Testing Guidelines
Write pytest modules as `test_<module>.py` and name cases `test_<behavior>__<condition>()`. Cover each preprocessing branch, at least one optimizer happy path per runway configuration, and failure modes such as missing taxi distances. For data refreshes, add regression tests that compare representative CSV slices (head/tail hashes) with fixtures to catch drift early.

## Commit & Pull Request Guidelines
Follow `<area>: <imperative summary>` for commit subjects (e.g., `models: relax towing penalty`) and include bodies referencing dataset versions or task IDs when relevant. Pull requests must describe motivation, enumerate major changes, link tracking issues, and attach evidence—console excerpts, metrics deltas, or solver traces. Request review from a data owner whenever `data/raw/` or preprocessing logic changes.

## Data & Security Notes
Never upload personally identifiable passenger details or supplier originals. Treat `data/raw/` as read-only provenance; share only sanitized derivatives in `data/clean/`. When publishing artifacts, provide `md5` checksums so downstream teams can verify integrity, and keep credentials or airport ops documents in external secret stores.
