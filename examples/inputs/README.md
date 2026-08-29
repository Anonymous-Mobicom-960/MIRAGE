# Example inputs

Deliberately empty.

The footage used for the reported results shows real people and is not redistributed. Supply your own
clip: any MP4 containing one or more people works.

**Expected format**

| Property | Value |
|---|---|
| Container / codec | MP4 / H.264 |
| Frame rate | **30 fps**; the cloud graph and the phone app both assume it (30 fps in, 30 fps out) |
| Resolution | Read at runtime from the artifact; the reported runs used 1264 x 1264 |
| Colour | 8-bit BGR after decode (`yuv420p` on disk) |

Prepare a clip with:

```bash
ffmpeg -i source.mp4 -ss <start> -t <duration> -vf "fps=30" \
       -c:v libx264 -crf 16 -pix_fmt yuv420p clip.mp4
```

Then follow [`../../workflows/end_to_end/RUNBOOK.md`](../../workflows/end_to_end/RUNBOOK.md).
