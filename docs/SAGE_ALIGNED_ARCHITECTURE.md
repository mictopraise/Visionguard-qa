# VisionGuard QA — Sage-Aligned Architecture

## Product contract

VisionGuard QA evaluates two videos using the same sequence a human quality rater would follow:

1. Assess Video A independently.
2. Assess Video B independently.
3. Assign a 1–5 MOS quality score to each video.
4. Assign artifact labels to each video.
5. Assess playback smoothness / temporal quality for each video.
6. Compare the two videos side-by-side after synchronization.
7. Decide whether Video A is better, Video B is better, they are the same, or the evidence is inconclusive.
8. Preserve evidence and agent trace for review.

Video A and Video B are peers. Neither is assumed to be the reference or ground truth.

## Human-aligned outputs

### MOS quality score

- 1 — Bad
- 2 — Poor
- 3 — Fair
- 4 — Visible level of one artifact
- 5 — No perceivable quality issue

MOS scoring must remain calibration-backed. The system must not manufacture a MOS result from unvalidated thresholds.

### Artifact taxonomy

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

Temporal issues include genuine playback irregularities such as skipping/jumping frames and flickering. Buffering is not treated as a video artifact.

### Pairwise preference

- Video A better
- Video B better
- Same
- Inconclusive

Preference should be consistent with each video's independent quality assessment and playback quality.

## Architecture

```text
Video A                         Video B
   |                               |
   v                               v
Spatial QA                      Spatial QA
- blur                          - blur
- blockiness                    - blockiness
- oversharpening/ringing        - oversharpening/ringing
- oversmoothing                 - oversmoothing
- color shift                   - color shift
- unnatural-object evidence     - unnatural-object evidence
   |                               |
   v                               v
Temporal QA                     Temporal QA
- freeze/repeated frames        - freeze/repeated frames
- motion discontinuity          - motion discontinuity
- flicker                       - flicker
- scene/edit context            - scene/edit context
   |                               |
   v                               v
Independent quality record      Independent quality record
- MOS 1–5                       - MOS 1–5
- artifacts                     - artifacts
- playback quality              - playback quality
   \                               /
    \                             /
     v                           v
       synchronized comparison
                |
                v
       A better / B better /
       same / inconclusive
                |
                v
       evidence-backed rationale
```

## Role of OpenCV and the agent

**OpenCV produces measurable visual evidence. The agent decides what analysis should happen next and how evidence should be reconciled.**

Examples:

- OpenCV detects low edge energy and detail loss; the agent asks whether the pattern is consistent with blur versus intentional depth-of-field.
- OpenCV detects high-frequency edge overshoot; the agent requests a ringing/oversharpening confirmation pass.
- OpenCV detects temporal instability; the agent checks whether the event overlaps a scene transition before treating it as a playback defect.
- Pairwise comparison finds a slight quality difference; the agent reconciles that preference with both independent MOS/artifact records.

## Existing modules retained

The following are retained as the Temporal QA branch:

- frame alignment
- freeze/repeated-frame detection
- motion analysis
- flicker detection
- scene-change context
- clustering
- targeted second-pass confirmation
- evidence export
- agent trace

## Milestone 6 — Human-Aligned Video Quality Scoring

Implement in this order:

1. Independent quality-assessment pipeline per video.
2. Blur detector.
3. Blockiness detector.
4. Oversharpening/ringing detector.
5. Oversmoothing/detail-loss detector.
6. Color-shift detector.
7. Connect existing temporal QA to playback-quality output.
8. Produce artifact labels per video.
9. Calibrate MOS scoring using controlled benchmark examples.
10. Produce pairwise preference that is consistent with independent scores.
11. Surface evidence and confidence in the UI.

## Guardrails

- Do not infer quality from semantic subject matter.
- Do not mark intentional blur as a defect merely because blur exists.
- Do not force a pairwise winner when evidence is weak.
- Do not let scene cuts alone cause a quality failure.
- Do not report calibrated MOS scores until benchmark evidence supports the thresholds.
- Keep detector measurements, agent reasoning, and human-facing labels separate so each can be audited.
