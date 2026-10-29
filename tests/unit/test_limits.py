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


def test_idle_keys_are_dropped_when_the_map_doubles() -> None:
    limiter = SlidingWindowLimiter(max_units=1, window_seconds=1)
    for i in range(8192):
        limiter.allow(f"old-{i}", now=0)
    # The map has reached its doubling threshold and every earlier client is idle, so this call's
    # sweep drops them all before the new key is added.
    assert limiter.allow("fresh", now=100)
    assert len(limiter._hits) == 1


def test_new_keys_are_refused_past_the_cap() -> None:
    limiter = SlidingWindowLimiter(max_units=1, window_seconds=60, max_keys=2)
    assert limiter.allow("a", now=0)
    assert limiter.allow("b", now=0)
    assert not limiter.allow("c", now=0)
    # A key already tracked is still served, so the cap does not lock out known clients.
    assert not limiter.allow("a", now=1)


def test_a_full_map_recovers_once_clients_go_idle() -> None:
    # Regression: the doubling threshold used to sit above the cap, so a full map of idle clients
    # was never swept and new clients stayed refused until the process restarted.
    limiter = SlidingWindowLimiter(max_units=1, window_seconds=10, max_keys=2)
    assert limiter.allow("a", now=0)
    assert limiter.allow("b", now=0)
    assert not limiter.allow("c", now=1)
    assert limiter.allow("c", now=11)


def test_exhausted_reports_without_spending_budget() -> None:
    limiter = SlidingWindowLimiter(max_units=1, window_seconds=10)
    assert not limiter.exhausted("a", now=0)
    assert limiter.allow("a", now=0)
    assert limiter.exhausted("a", now=1)
    assert not limiter.exhausted("a", now=10)


def test_ipv6_clients_in_one_slash64_share_a_budget() -> None:
    assert client_key("2001:db8::1") == client_key("2001:db8::ffff")
    assert client_key("2001:db8:1::1") != client_key("2001:db8::1")


def test_ipv4_and_unparseable_hosts_are_used_as_given() -> None:
    assert client_key("10.0.0.1") == "10.0.0.1"
    assert client_key("testclient") == "testclient"
