# Shared scripts

Small utilities that more than one tier depends on.

```text
scripts/
└── utilities/
    └── console_safe.py
```

## `utilities/console_safe.py`

`import console_safe`. That is the whole interface.

Python on Windows writes stdout in the console code page, and any character outside it raises
`UnicodeEncodeError`. The status lines in this project use `✅ / 🔴 / ⚠️ / → / ±`, so on such a console
a script does its work correctly and then **dies on the summary print**, which reads as a failed
run. The module reconfigures `stdout`/`stderr` to UTF-8 with `errors="replace"` only when the stream
cannot already encode those characters, so on a UTF-8 terminal it is a no-op and nothing is silenced.

It asks the encoder whether it can encode; it never probes by writing to the stream, because the
first version did exactly that and emitted stray glyphs into real output on terminals that could.

Model weights are not downloaded by a script here; they are documented instead, per model, in
[`../models/README.md`](../models/README.md), because several of them cannot be fetched
non-interactively and several carry licences a user must read before accepting.
