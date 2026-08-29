# Licence: not yet selected

**No licence has been chosen for this repository.** Until one is added, no rights are granted beyond
viewing this material. Do not assume MIT, Apache-2.0, BSD or any other permissive default.

No licence file was present anywhere in the source project this release was curated from, so
selecting one would have been a decision the curator is not authorised to make. It has to be made by
the authors, before the repository is made public.

## What the choice is constrained by

The choice is **not free**. Three third-party components propagate obligations into any work that
combines with them:

| Component | Licence | Consequence |
|---|---|---|
| Ultralytics YOLO11 / YOLO11n-seg (the person detector and segmenter) | **AGPL-3.0** | Strong copyleft, with a network-use clause. A derived work that combines with it cannot be released under a permissive licence, and network use triggers the source-provision obligation. This is the binding constraint. |
| `big-lama` inpainting weights | **CC BY-NC-SA 4.0** | Non-commercial and share-alike. The LaMa *code* is Apache-2.0; the *weights* are not. |
| InsightFace models (`genderage.onnx`, `buffalo_l`) | **Non-commercial research use only** | Bars commercial use of any deployment that includes them. |

Two consequences follow, and both should be settled deliberately rather than discovered later:

1. **A permissive licence on this repository would be inconsistent with the AGPL-3.0 detector** if the
   two are treated as a combined work. Either adopt a compatible copyleft licence, or make the
   detector a genuinely separable, user-supplied dependency and say so explicitly.
2. **No weights are redistributed here**, which keeps the weight licences off this repository's own
   terms, but they still bind anyone who *runs* the system. That is documented in
   [`THIRD_PARTY.md`](THIRD_PARTY.md) and [`models/README.md`](models/README.md).

## 🔴 The separability question is now settled, and the answer is "not separable" (measured 2026-08-28)

Point 1 below offered two ways out: adopt compatible copyleft, **or** make the AGPL-3.0 detector a
genuinely separable, user-supplied dependency. The second option was checked against the code
rather than assumed, and it is **not currently available**:

* `tier1/src/mirage/pipeline.py:27-28` imports `YOLOSegBlur` and `YOLO11nBoxBlur` at module
  level; they are the shipped Tier-1 segmentation and detection backends.
* Both call `from ultralytics import YOLO` and construct a `YOLO(...)` object
  (`blur_yoloseg.py:38`, `blur_yolo11n.py:38`).
* **The ONNX route does not avoid it.** `blur_yoloseg.py:38,71` uses `ultralytics.YOLO` to load the
  `.onnx` *and* to export one from the `.pt` when it is missing. So selecting ONNX changes the
  execution graph, not the dependency.

The user supplies the *weights*, but the *code path* runs through AGPL-3.0 software on every
configuration the repository ships. Anyone who wants a permissive licence would first have to
replace that call site with a direct ONNX Runtime session -- which `pipeline.py:1296` already notes
as possible but out of scope -- and re-validate the masks, since the mask is the privacy boundary.

This does not choose the licence. It removes the ambiguity that was blocking the choice: the
"separable dependency" route requires code changes that have not been made.

## Before publication

- [ ] Choose a licence, taking the above into account, and confirm it with every author's institution.
- [ ] Add the licence text as `LICENSE` at the repository root and delete this file.
- [ ] Set the `license` field in [`CITATION.cff`](CITATION.cff).
- [ ] Confirm that the chosen licence is compatible with every entry in
      [`THIRD_PARTY.md`](THIRD_PARTY.md).
- [ ] Decide whether the AGPL-3.0 detector is a combined work or a separable dependency, and state
      that decision in the README.
