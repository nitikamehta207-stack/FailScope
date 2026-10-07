"""
sandbox.py - the "production system" the agent is allowed to fix.

It owns a running incident (the fault config is private, the agent never reads it).
The agent only gets:
    sandbox.store            -> telemetry of the incident (days 0..29)
    sandbox.apply_fix(name)  -> telemetry AFTER that fix was applied (days 0..39)

How apply_fix works: the same seed gives the same data, so we simply replay the
same world, but with the fix switched on from day 30. Days 0..29 are identical
to the incident; days 30..39 show what happens after the fix.
"""
from .simulator import Simulator


class Sandbox:
    def __init__(self, sim=None, fault=None, seed=0, variant="unit",
                 n_days=30, fault_day=15):
        self.sim = sim or Simulator()
        self._cfg = dict(fault=fault, seed=seed, variant=variant, fault_day=fault_day)
        self.fix_day = n_days                      # the fix is applied right after the incident window
        self.store = self.sim.run(n_days=n_days, **self._cfg)

    def apply_fix(self, action, observe_days=10):
        """Apply `action` on fix_day and return the new telemetry store."""
        return self.sim.run(n_days=self.fix_day + observe_days,
                            fix=action, fix_day=self.fix_day, **self._cfg)

    @property
    def truth(self):
        """Ground truth: ONLY for scoring in evaluation code, never for the agent."""
        return self.store.ground_truth