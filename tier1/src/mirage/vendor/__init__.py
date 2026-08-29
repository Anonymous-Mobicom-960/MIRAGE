"""Third-party / upstream code vendored into this tree.

HARD RULE (owner decision): every module under this package is a BYTE-IDENTICAL lift of code that
lives and is measured somewhere else. Nothing here is ever edited, reformatted, linted or
"cleaned up" -- the measured behaviour is a property of the exact bytes. All host-specific logic
belongs in an adapter OUTSIDE this package.

Each vendored module documents, at the top of the file: the source repo + commit, the source file,
and the exact line range it was lifted from, so the lift can be re-verified with a byte diff.
"""
