#!/usr/bin/env python3
"""Make `print` survive this machine's cp1252 console. Import it; that is the whole interface.

    import console_safe   # noqa: F401

WHY THIS EXISTS. Python on Windows writes stdout in the console codepage - cp1252 here - and any
character outside it raises `UnicodeEncodeError`. Every ✅/🔴/⚠️/→ in a status line is such a
character, and this repo is full of them.

The failure mode is nastier than a normal crash, which is why it kept being missed:

  * It fires LATE. `requeue_c2fix.py` ran its whole gate, printed three correct result lines, and
    then died on the summary - so the run LOOKED failed while the work had actually succeeded.
  * It fires ONLY on the path that reaches that print. A script can pass every rehearsal and crash
    the first time a condition is true.
  * It is invisible to review: the source is valid UTF-8 and reads fine.

It has now bitten three separate times in one session - a driver draft that had already found its
button, `requeue_c2fix.py`, and a sweep that found NINE more scripts one import away from the same
death, several on the critical path (`check_render.py`, `finish_clip.py`, `alpha_from_tier1.py`).

WHAT IT DOES. Reconfigures stdout/stderr to UTF-8 with `errors="replace"` when the stream cannot
already encode the characters we use. Nothing is silenced: a character that will not fit is replaced
with `?` and the line still prints. If the terminal is already UTF-8 (Windows Terminal, a redirect
to a file, CI) this is a no-op and the emoji render properly.

Deliberately NOT done: stripping the emoji from the source. They carry meaning at a glance in these
status lines, and the defect is the ENCODER, not the characters.
"""
import sys



def _fix(stream):
    if stream is None:
        return
    # 🔴 Do NOT probe by writing to the stream. The first version of this file did exactly that - 
    # `stream.write("✅")` - and on a console that COULD encode it, the probe character was emitted
    # into real output, so every patched script began its run by printing stray glyphs. Caught by
    # running it rather than reasoning about it. Ask the ENCODER whether it can encode; never make
    # the terminal answer.
    enc = getattr(stream, "encoding", None)
    if enc:
        try:
            "✅🔴⚠️→±·".encode(enc)
            return                        # the stream copes; leave it completely alone
        except (UnicodeEncodeError, LookupError):
            pass
    try:
        # Python 3.7+: retarget the existing buffer rather than replacing the object, so anything
        # already holding a reference to sys.stdout keeps working.
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        try:
            import io
            enc = getattr(stream, "encoding", None) or "utf-8"
            wrapped = io.TextIOWrapper(stream.buffer, encoding=enc, errors="replace",
                                       line_buffering=True)
            if stream is sys.stdout:
                sys.stdout = wrapped
            elif stream is sys.stderr:
                sys.stderr = wrapped
        except Exception:
            pass                          # never let a logging nicety take a render down


_fix(sys.stdout)
_fix(sys.stderr)
