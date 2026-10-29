"""Unit tests for the rate-limit budget: per-unit cost, window expiry, IPv6 keying and cleanup."""

from __future__ import annotations

from defectlens.serving.limits import SlidingWindowLimiter, client_key


def test_cost_is_charged_per_unit() -> None:
    limiter = SlidingWindowLimiter(max_units=4, window_seconds=60)
    assert limiter.allow("a", cost=3, now=0)
    assert not limiter.allow("a", cost=2, now=1)
    assert limiter.allow("a", cost=1, now=2)


def test_window_expires_old_units() -> None:
    limiter = SlidingWindowLimiter(max_units=2, window_seconds=10)
    assert limiter.allow("a", now=0)
    assert limiter.allow("a", now=1)
    assert not limiter.allow("a", now=5)
    assert limiter.allow("a", now=10)


def test_idle_keys_are_dropped_once_the_map_is_large() -> None:
    limiter = SlidingWindowLimiter(max_units=1, window_seconds=1)
    for i in range(5000):
        limiter.allow(f"client-{i}", now=0)
    # Every earlier client is idle by now, so the sweep this call triggers drops them all.
    assert limiter.allow("fresh", now=100)
    assert len(limiter._hits) == 1


def test_ipv6_clients_in_one_slash64_share_a_budget() -> None:
    assert client_key("2001:db8::1") == client_key("2001:db8::ffff")
    assert client_key("2001:db8:1::1") != client_key("2001:db8::1")


def test_ipv4_and_unparseable_hosts_are_used_as_given() -> None:
    assert client_key("10.0.0.1") == "10.0.0.1"
    assert client_key("testclient") == "testclient"
