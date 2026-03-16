from __future__ import annotations

class ScratchRepresentation:
    def __init__(self) -> None:
        self.previous = None
        self.next = None
        self.contained = []
        self.template_path = "templates/none"
        self.constants = []
        self.variables = []

    def attach_previous(self, previous: ScratchRepresentation) -> None:
        if self.previous is not None:
            self.previous.next = None
        self.previous = previous
        previous.next = self

    def attach_next(self, next: ScratchRepresentation):
        if self.next is not None:
            self.next.previous = None
        self.next = next
        next.previous = self