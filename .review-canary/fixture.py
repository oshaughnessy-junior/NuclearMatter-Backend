"""Disposable arithmetic fixture; never imported by the scientific package."""


def weighted_mean(values, weights):
    """Return the weighted mean for the documented valid input domain."""
    return sum(value * weight for value, weight in zip(values, weights)) / len(values)
