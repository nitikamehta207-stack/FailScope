"""
evidence.py - the one standard "report card" every tool returns.

The agent never sees raw tables. It only sees Evidence objects, and every
claim it makes later must point to an Evidence id.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class Evidence:
    id: str            # "E1", "E2", ... (unique per investigation)
    layer: str         # "data" | "model" | "system"
    tool: str          # which tool produced it
    finding: str       # one plain-English sentence
    data: dict = field(default_factory=dict)   # the raw numbers behind the finding
    anomalous: bool = False                    # did this tool see something abnormal?

    def to_dict(self):
        return asdict(self)

    def short(self):
        flag = "ANOMALY" if self.anomalous else "normal "
        return f"[{self.id}] {self.layer:<6} {flag} {self.tool}: {self.finding}"