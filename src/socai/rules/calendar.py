"""Lịch âm–dương từ `calendar.csv` của split (cột: date, weekday, lunar_day, lunar_month, lunar_year,
lunar_leap, holiday) và cửa sổ Tết theo rulebook: 23 tháng Chạp năm âm Y−1 → hết mùng 10 Giêng năm Y."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


@dataclass
class LunarCalendar:
    rows: dict[date, tuple[int, int, int, bool, str]]  # ngày dương → (ngày âm, tháng âm, năm âm, nhuận, lễ)

    @classmethod
    def load(cls, path: Path) -> LunarCalendar:
        rows = {}
        with path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rows[date.fromisoformat(r["date"])] = (
                    int(r["lunar_day"]), int(r["lunar_month"]), int(r["lunar_year"]),
                    r["lunar_leap"].strip().lower() in ("true", "1"), r.get("holiday") or "")
        return cls(rows)

    def solar(self, lunar_year: int, lunar_month: int, lunar_day: int, leap: bool = False) -> date | None:
        for d, (ld, lm, ly, lp, _) in self.rows.items():
            if (ly, lm, ld, lp) == (lunar_year, lunar_month, lunar_day, leap):
                return d
        return None

    def tet_window(self, year: int, start: tuple[int, int] = (12, 23), end: tuple[int, int] = (1, 10)
                   ) -> tuple[date, date] | None:
        """Tết năm `year`: từ (tháng, ngày) âm `start` của năm âm year−1 đến hết `end` của năm âm year."""
        a = self.solar(year - 1, start[0], start[1])
        b = self.solar(year, end[0], end[1])
        return (a, b) if a and b else None

    def holidays(self, start: date, end: date) -> dict[date, str]:
        d, out = start, {}
        while d <= end:
            if d in self.rows and self.rows[d][4]:
                out[d] = self.rows[d][4]
            d += timedelta(days=1)
        return out
