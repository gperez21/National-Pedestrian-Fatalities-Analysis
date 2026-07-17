"""
dst.py — US Daylight Saving Time transition date calculations.

Computes spring-forward and fall-back dates for a given year, applying the
two US rule regimes:

* Pre-2007:  spring = 1st Sunday of April;  fall = last Sunday of October.
* 2007-present: spring = 2nd Sunday of March; fall = 1st Sunday of November.
  (Energy Policy Act of 2005, in effect from 2007.)
"""

from __future__ import annotations

from datetime import date, timedelta

# Python weekday integer for Sunday (Monday=0 … Sunday=6).
_SUNDAY = 6



def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the *n*th occurrence of *weekday* in *year*/*month*.

    Args:
        year: Four-digit year.
        month: Month number (1–12).
        weekday: Python weekday number (0=Mon … 6=Sun).
        n: 1-indexed position (1 = first, 2 = second, …).

    Returns:
        ``date`` of the nth weekday in that month.
    """
    first = date(year, month, 1)
    days_ahead = (weekday - first.weekday()) % 7
    first_occurrence = first + timedelta(days=days_ahead)
    return first_occurrence + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of *weekday* in *year*/*month*.

    Args:
        year: Four-digit year.
        month: Month number (1–12).
        weekday: Python weekday number (0=Mon … 6=Sun).

    Returns:
        ``date`` of the last such weekday in that month.
    """
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    days_back = (last.weekday() - weekday) % 7
    return last - timedelta(days=days_back)


def dst_transition_dates(year: int) -> tuple[date, date]:
    """Return the (spring-forward, fall-back) DST transition dates for *year*.

    Applies US DST schedule:

    * **Pre-2007**: spring-forward on the 1st Sunday of April;
      fall-back on the last Sunday of October.
    * **2007 onward**: spring-forward on the 2nd Sunday of March;
      fall-back on the 1st Sunday of November.

    Args:
        year: Four-digit year (FARS coverage starts 1975; DST rules predate it).

    Returns:
        Tuple ``(spring_forward, fall_back)`` as ``datetime.date`` objects.

    Examples:
        >>> dst_transition_dates(2006)
        (datetime.date(2006, 4, 2), datetime.date(2006, 10, 29))
        >>> dst_transition_dates(2007)
        (datetime.date(2007, 3, 11), datetime.date(2007, 11, 4))
        >>> dst_transition_dates(2023)
        (datetime.date(2023, 3, 12), datetime.date(2023, 11, 5))
    """
    if year >= 2007:
        spring = _nth_weekday(year, 3, _SUNDAY, 2)   # 2nd Sunday of March
        fall   = _nth_weekday(year, 11, _SUNDAY, 1)  # 1st Sunday of November
    else:
        spring = _nth_weekday(year, 4, _SUNDAY, 1)   # 1st Sunday of April
        fall   = _last_weekday(year, 10, _SUNDAY)    # last Sunday of October

    return spring, fall
