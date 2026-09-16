# VisionGuard QA

VisionGuard QA is an agentic visual quality-assurance system for comparing and evaluating short videos.

It uses OpenCV 5 to detect temporal and visual anomalies, extract structured evidence, and support an agentic decision loop that determines what analysis should happen next.

## Core Principle

**OpenCV produces the visual evidence. The agent decides what happens next.**

## Planned MVP

VisionGuard QA will compare Video A and Video B and analyze:

- frame alignment
- frozen or repeated frames
- motion discontinuities
- scene inconsistencies
- flicker
- subject or region persistence

Detected issues will be converted into structured evidence containing:

- issue type
- timestamps
- confidence
- measurements
- evidence frames

The agent layer can then choose to:

- PASS
- RECHECK
- REGENERATE
- HUMAN REVIEW

## Agentic Vision Loop

1. OpenCV performs an initial visual comparison.
2. A possible anomaly is detected.
3. The agent selects an appropriate deeper visual check.
4. OpenCV runs that targeted analysis.
5. Evidence is gathered.
6. VisionGuard produces a final QA disposition.

## Planned Stack

- Python
- OpenCV 5
- FastAPI
- NumPy
- Pydantic
- AWS
- Amazon S3
- Amazon ECS
- Docker
- HTML / CSS / JavaScript

## Evaluation Plan

The competition benchmark will use controlled short-video pairs with known visual defects such as:

- repeated frames
- frozen sections
- temporary blackouts
- motion jumps
- brightness flicker
- scene inconsistencies

Evaluation will measure:

- precision
- recall
- F1 score
- false-positive rate
- temporal localization accuracy
- agent next-action accuracy
- final QA disposition accuracy
- processing time

## Local Development

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API.

Run tests with:

```bash
pytest
```

The `/health` endpoint reports the installed OpenCV version. The current foundation intentionally asserts OpenCV 5.x.

## Status

**Milestone 1 — Foundation: IN PROGRESS**

Implemented:
- FastAPI application shell
- OpenCV 5 dependency
- health/version endpoint
- Video A / Video B upload ingestion
- OpenCV metadata probing
- foundational data models
- Docker deployment scaffold
- first automated test

Next:
- frame normalization and alignment
- controlled defect generator
- frozen/repeated-frame detector

## Competition

OpenCV AI Competition 2026 powered by AWS

Special Award target: Agentic Vision Award
