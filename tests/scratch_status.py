"""Status values that Scratch projects read from ``cattorch status``.

They are the public Scratch contract (docs/scratch-interface.md), so tests
write them out here, once, instead of taking them from ``cattorch.interface``:
a change there must fail the tests, not follow them.
"""

OVERTAKEN = "another call ran during this one"
INTERRUPTED = "an earlier call was stopped or is still running; reset first"
