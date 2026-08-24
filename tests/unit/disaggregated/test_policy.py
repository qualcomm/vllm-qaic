# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import itertools
import threading

import pytest

pytest.importorskip(
    "qaic_disagg", reason="qaic_disagg is an optional dependency, not installed here"
)

from qaic_disagg.proxy.server import (
    LeastOutstandingSchedulingPolicy,
    RoundRobinSchedulingPolicy,
)


class TestRoundRobinSchedulingPolicy:
    def test_schedule_basic_round_robin(self):
        policy = RoundRobinSchedulingPolicy()
        instances = ["inst1", "inst2", "inst3"]
        cycler = itertools.cycle(instances)

        # First pass
        assert policy.schedule(cycler) == "inst1"
        assert policy.schedule(cycler) == "inst2"
        assert policy.schedule(cycler) == "inst3"
        # Second pass
        assert policy.schedule(cycler) == "inst1"
        assert policy.schedule(cycler) == "inst2"


class TestLeastOutstandingSchedulingPolicy:
    @pytest.fixture
    def policy_and_instances(self):
        policy = LeastOutstandingSchedulingPolicy()
        instances = ["server_a", "server_b", "server_c"]
        return policy, instances

    def test_schedule_initial_distribution(self, policy_and_instances):
        policy, instances = policy_and_instances

        # Initial state: all instances have 0 outstanding
        assert (
            policy.schedule(None, instances) == "server_a"
        )  # First choice (alphabetical/insertion order)
        assert policy.instance_usage_outstanding["server_a"] == 1
        assert policy.instance_usage_outstanding.get("server_b", 0) == 0
        assert policy.instance_usage_outstanding.get("server_c", 0) == 0

        assert policy.schedule(None, instances) == "server_b"  # Next least outstanding
        assert policy.instance_usage_outstanding["server_a"] == 1
        assert policy.instance_usage_outstanding["server_b"] == 1
        assert policy.instance_usage_outstanding.get("server_c", 0) == 0

        assert policy.schedule(None, instances) == "server_c"  # Next least outstanding
        assert policy.instance_usage_outstanding["server_a"] == 1
        assert policy.instance_usage_outstanding["server_b"] == 1
        assert policy.instance_usage_outstanding["server_c"] == 1

        assert (
            policy.schedule(None, instances) == "server_a"
        )  # All are 1, so back to server_a (min by key)
        assert policy.instance_usage_outstanding["server_a"] == 2
        assert policy.instance_usage_outstanding["server_b"] == 1
        assert policy.instance_usage_outstanding["server_c"] == 1

    def test_post_schedule_update_decrements_count(self, policy_and_instances):
        policy, instances = policy_and_instances

        # Schedule a few to build up outstanding counts
        policy.schedule(None, instances)  # server_a: 1
        policy.schedule(None, instances)  # server_b: 1
        policy.schedule(None, instances)  # server_c: 1
        policy.schedule(None, instances)  # server_a: 2

        assert policy.instance_usage_outstanding["server_a"] == 2
        assert policy.instance_usage_outstanding["server_b"] == 1
        assert policy.instance_usage_outstanding["server_c"] == 1

        # Now, simulate a request completion
        policy.post_schedule_update("server_a")
        assert policy.instance_usage_outstanding["server_a"] == 1
        assert policy.instance_usage_outstanding["server_b"] == 1
        assert policy.instance_usage_outstanding["server_c"] == 1

        policy.post_schedule_update("server_b")
        assert policy.instance_usage_outstanding["server_a"] == 1
        assert policy.instance_usage_outstanding["server_b"] == 0
        assert policy.instance_usage_outstanding["server_c"] == 1

        # server_b should be picked next
        assert policy.schedule(None, instances) == "server_b"
        assert policy.instance_usage_outstanding["server_b"] == 1

    def test_post_schedule_update_non_existent_instance(self, policy_and_instances):
        policy, instances = policy_and_instances

        # Schedule some requests first
        policy.schedule(None, instances)
        policy.schedule(None, instances)

        # Attempt to update a non-existent instance, should not raise error
        try:
            policy.post_schedule_update("non_existent_server")
        except Exception as e:
            pytest.fail(
                f"post_schedule_update raised an unexpected exception for non-existent instance: {e}"
            )

        # Verify existing counts are unaffected
        assert policy.instance_usage_outstanding["server_a"] == 1
        assert policy.instance_usage_outstanding["server_b"] == 1

    def test_concurrent_scheduling(self, policy_and_instances):
        policy, instances = policy_and_instances
        num_threads = 10
        num_requests_per_thread = 100
        total_requests = num_threads * num_requests_per_thread

        def worker():
            for _ in range(num_requests_per_thread):
                instance = policy.schedule(None, instances)
                # In a real scenario, this would be `policy.post_schedule_update(instance)`
                # after the request completes. For this test, we just want to verify counts.
                # We'll call post_schedule_update later to ensure final counts are zero.

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # After all scheduling, the sum of outstanding should be total_requests
        sum_outstanding = sum(policy.instance_usage_outstanding.values())
        assert sum_outstanding == total_requests

        # All instances should have roughly equal outstanding requests
        avg_outstanding_per_instance = total_requests / len(instances)
        for inst in instances:
            # Allow for a small deviation due to non-deterministic thread scheduling
            # and the nature of LO (it tries to balance, but not perfectly in real-time
            # without completion signals).
            assert (
                abs(
                    policy.instance_usage_outstanding[inst]
                    - avg_outstanding_per_instance
                )
                < len(instances) * 2
            )

        # Simulate all requests completing
        for _ in range(total_requests):
            # Find an instance to decrement
            instance_to_decrement = None
            for inst, count in policy.instance_usage_outstanding.items():
                if count > 0:
                    instance_to_decrement = inst
                    break
            if instance_to_decrement:
                policy.post_schedule_update(instance_to_decrement)
            else:
                break  # All are 0, no more to decrement

        assert sum(policy.instance_usage_outstanding.values()) == 0

    def test_schedule_single_instance(self, policy_and_instances):
        policy, _ = policy_and_instances
        instances = ["single_server"]

        assert policy.schedule(None, instances) == "single_server"
        assert policy.instance_usage_outstanding["single_server"] == 1

        policy.post_schedule_update("single_server")
        assert policy.instance_usage_outstanding["single_server"] == 0

        assert policy.schedule(None, instances) == "single_server"
        assert policy.instance_usage_outstanding["single_server"] == 1
