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

Spatial QA is now underway.

Implemented in the current Milestone 6 build:

- independent spatial sampling per video
- blur evidence using Laplacian detail energy
- blockiness evidence using 8-pixel boundary discontinuity measurements
- oversharpening/ringing evidence using high-frequency edge residual and overshoot measurements
- oversmoothing/detail-loss evidence using fine-to-coarse texture energy
- color-shift evidence using persistent CIELAB chroma-bias measurements
- provisional candidate artifact output per video
- UI display of spatial candidates separately from temporal verdicts
- synthetic tests for blur sensitivity and block-boundary detection

Important: blur, blockiness, oversharpening, oversmoothing, and color-shift thresholds are **provisional calibration thresholds**, not final Sage labels. Candidate artifacts do not yet force MOS scores or final quality verdicts.

Milestone 6E now adds:
- a controlled synthetic calibration benchmark for all five implemented spatial signals
- 12 clean/degraded sample pairs per artifact class
- directional separation reporting for each signal
- a calibration registry that blocks final artifact labels until promotion criteria are met
- CI execution and report export for spatial calibration

Remaining Milestone 6 work:

1. inspect and freeze benchmark-backed thresholds
2. validate against human/example-labeled clips, not synthetic data alone
3. promote reliable candidates into final artifact labels
4. calibrate MOS scoring
5. produce pairwise preference consistent with both independent assessments
6. surface final evidence and confidence in the UI

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

**Architecture revision: frozen and CI-verified**

VisionGuard now follows independent quality evaluation first, synchronized comparison second.

**Milestone 6E: COMPLETE — controlled spatial calibration passed for all five signals; registry marks them synthetic-validated, while final artifact promotion remains blocked pending human/example-labeled calibration.**

## Competition

OpenCV AI Competition 2026 powered by AWS

Special Award target: Agentic Vision Award


## Milestone 6E Synthetic Calibration Result

CI run 35356430169 validated 12 clean/degraded pairs per spatial artifact signal.

- Blur: clean median 2942.19 vs degraded 1.90; separation ratio 1547.71×
- Blockiness: clean median 0.98 vs degraded 7.93; separation ratio 8.12×
- Oversharpening: clean median 15.64 vs degraded 31.39; separation ratio 2.01×
- Oversmoothing: clean median 0.744 vs degraded 0.584; separation ratio 1.27×
- Color shift: clean median 6.08 vs degraded 42.56; separation ratio 7.00×

All five signals moved in the expected direction. These results validate detector behavior on controlled synthetic degradations only; they do not establish human-level classification accuracy.
