"""Shared test setup.

The API's rate limiter is a process-global with a 60-second wall-clock window,
so without a reset the whole suite shares one budget and a later test can see a
spurious 429. Clear it before every test.
"""
import pytest

from api.auth import _limiter


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    _limiter.reset()
    yield
