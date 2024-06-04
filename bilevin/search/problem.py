from typing import TYPE_CHECKING

from domains.domain import Domain
if TYPE_CHECKING:
    pass


class Problem:
    def __init__(self, id: int, domain: Domain):
        self.id: int = id
        self.domain: Domain = domain

    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        return self.id == other.id
