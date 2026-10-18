from copy import deepcopy
from dataclasses import dataclass, field
import uuid


def initial_style():
    return {'kind': 'blur', 'strength': 1., 'padding': .25, 'color': '#202020'}


@dataclass
class DocumentState:
    kind: str
    asset: dict | None = None
    regions: list = field(default_factory=list)
    seeds: list = field(default_factory=list)
    analysis: dict | None = None
    style: dict = field(default_factory=initial_style)
    identity: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_path: str | None = None
    dirty: bool = False
    edit_revision: int = 0
    region_undo: list = field(default_factory=list)

    def mark_edited(self):
        self.edit_revision += 1
        self.dirty = True

    def remember_regions(self):
        self.region_undo.append(deepcopy(self.regions))
        del self.region_undo[:-50]

    def undo_regions(self):
        if not self.region_undo:
            return False
        self.regions = self.region_undo.pop()
        self.mark_edited()
        return True

    def snapshot(self):
        return deepcopy({'kind': self.kind, 'asset': self.asset,
                         'regions': self.regions, 'seeds': self.seeds,
                         'analysis': self.analysis, 'style': self.style})

    @classmethod
    def from_project(cls, project):
        return cls(project['kind'], asset=deepcopy(project['asset']),
                   regions=deepcopy(project.get('regions') or []),
                   seeds=deepcopy(project.get('seeds') or []),
                   analysis=deepcopy(project.get('analysis')),
                   style=deepcopy(project.get('style') or initial_style()))
