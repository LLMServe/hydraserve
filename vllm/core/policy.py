from collections import deque
from typing import Deque

from vllm.sequence import SequenceGroup


class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            ))


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        # place running requests at the end of waiting queue
        if seq_group.in_process:
            return 0
        prio = now - seq_group.metrics.arrival_time
        # prioritize cold start requests
        if not seq_group.use_dest:
            prio += 1000
        return prio


class PolicyFactory:

    _POLICY_REGISTRY = {'fcfs': FCFS}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
