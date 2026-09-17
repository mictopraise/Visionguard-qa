# VisionGuard QA

VisionGuard QA is an agentic visual quality-assurance system for evaluating two videos independently and then comparing their overall visual quality.

It uses OpenCV 5 to produce measurable spatial and temporal evidence, preserves that evidence for review, and supports an agentic decision loop that determines which deeper visual check should happen next.

## Core Principle

**OpenCV produces the visual evidence. The agent decides what happens next.**

## Product Workflow

VisionGuard follows a human-aligned video-quality workflow:

1. Evaluate Video A independently.
2. Evaluate Video B independently.
3. Assign a calibrated 1–5 MOS quality score to each video.
4. Assign artifact labels to each video.
5. Assess playback smoothness / temporal quality for each video.
6. Synchronize and compare the two videos.
7. Decide whether Video A is better, Video B is better, they are the same, or the evidence is inconclusive.
8. Preserve evidence frames, measurements, confidence, and the agent trace.

Video A and Video B are peers. Neither is assumed to be a reference or ground truth.

## Human-Facing Quality Outputs

### MOS score

- 1 — Bad
- 2 — Poor
- 3 — Fair
- 4 — Visible level of one artifact
- 5 — No perceivable quality issue

VisionGuard does **not** report a calibrated MOS until benchmark evidence supports the scoring thresholds.

### Artifact labels

- Unnatural objects
- Oversharpening
- Blurring
- Oversmooth
- Blocky
- Color shift
- Other
- No artifacts

### Playback quality

- Smooth
- Temporal issues
- Uncertain

### Pairwise preference

- Video A better
- Video B better
- Same
- Inconclusive

## Existing Temporal QA

Implemented OpenCV modules include:

- temporal alignment
- freeze / repeated-frame detection
- optical-flow motion analysis
- flicker detection
- scene-change context
- event clustering
- targeted second-pass confirmation
- aligned pairwise comparison
- evidence-frame export
- agent action trace

These modules form the **Temporal QA** branch of the broader product.

## Milestone 6 — Human-Aligned Video Quality Scoring

The next milestone adds the missing **Spatial QA** branch in this order:

1. independent quality-assessment pipeline per video
2. blur detector
3. blockiness detector
4. oversharpening / ringing detector
5. oversmoothing / detail-loss detector
6. color-shift detector
7. connect temporal QA to playback-quality output
8. produce artifact labels per video
9. calibrate MOS scoring on controlled examples
10. produce pairwise preference consistent with both independent assessments
11. surface evidence and confidence in the UI

The frozen architecture is documented in `docs/SAGE_ALIGNED_ARCHITECTURE.md`.

## Agentic Vision Loop

1. OpenCV performs an initial visual analysis.
2. A possible quality issue is detected.
3. The agent selects the appropriate deeper visual check.
4. OpenCV runs that targeted analysis.
5. Evidence is gathered and contextual checks are applied.
6. The result contributes to the video's independent quality record.
7. Synchronized A/B comparison reconciles the two records into an overall preference.

## Planned Stack

- Python 3.12
- OpenCV 5
- FastAPI
- NumPy
- Pydantic
- AWS
- Amazon S3
- Amazon ECS / Fargate
- Docker
- HTML / CSS / JavaScript

## Evaluation Plan

Controlled short-video pairs will be created with known defects and quality degradations. Evaluation will measure:

- artifact precision / recall / F1
- false-positive rate
- temporal localization accuracy
- playback-quality accuracy
- MOS agreement with labeled examples
- pairwise-preference accuracy
- agent next-action accuracy
- processing time

## Local Development

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` for the browser UI and `http://127.0.0.1:8000/docs` for the API.

Run tests with:

```bash
pytest
```

## Status

**Milestones 1–5: implemented**

- foundation / upload / probe
- baseline OpenCV temporal detectors
- clustering / evidence layer
- targeted second-pass confirmation
- temporal alignment + pairwise comparison

**Architecture revision: frozen**

VisionGuard now follows independent quality evaluation first, synchronized comparison second.

**Next: Milestone 6 — Human-Aligned Video Quality Scoring**

## Competition

OpenCV AI Competition 2026 powered by AWS

Special Award target: Agentic Vision Award
