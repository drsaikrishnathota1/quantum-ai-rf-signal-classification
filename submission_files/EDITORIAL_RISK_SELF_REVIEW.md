# Editorial Risk Self-Review

This checklist records the corrections made after the Defence Technology
rejection feedback on a separate manuscript. It is not meant for journal upload;
it is a local quality-control note.

## Rejection Pattern To Avoid

- Weak novelty framing.
- Thin literature survey.
- Overclaiming quantum or defense relevance.
- Submission files carrying inconsistent titles or stale abstracts.
- Results presented without enough ablation or robustness context.

## Corrections Applied To This Manuscript

- Reframed the paper as a reproducible robustness benchmark and ablation study.
- Expanded the literature base to 25 references covering classical AMC, deep
  AMC, RFML robustness/adversarial work, jamming, and quantum-kernel learning.
- Added a dedicated Related Work And Research Gap section.
- Clarified that the work does not claim quantum advantage.
- Added same-feature PCA-RBF SVM ablation against the simulated quantum kernel.
- Added computational complexity and latency reporting.
- Aligned title page, cover letter, highlights, manual submission script, README,
  and generated DOCX package builder to the same title and claim.
- Removed stale/overbroad framing that could read like a simple reapplication of
  an existing AI method.

## Submission Framing

Use this central claim:

Clean RF classification accuracy is not enough for contested-spectrum model
selection. A submission-grade evaluation should report clean performance,
degraded-condition robustness drops, ablation evidence, and computational cost.

Do not claim:

- quantum advantage;
- operational deployment readiness;
- classified defense validation;
- superiority over the full AMC literature.

