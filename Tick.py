from typing import Tuple
from datetime import datetime, timedelta


class Tick:
    def __init__(self, open_price: float, close_price: float, max_price: float, min_price: float, volume: float,
                 begin_time: datetime):
        self.open_price = open_price
        self.close_price = close_price
        self.max_price = max_price
        self.min_price = min_price
        self.volume = volume
        self.begin_time = begin_time

    @property
    def bullish(self) -> bool:
        return self.is_bullish()

    @property
    def bearish(self) -> bool:
        return self.is_bearish()

    def is_bullish(self) -> bool:
        return self.open_price < self.close_price

    def is_bearish(self) -> bool:
        return self.open_price > self.close_price

    def in_period(self, timestamp: datetime) -> bool:
        return False

    def add_trade(self, trade_volume: float, trade_price: float) -> None:
        pass

    @classmethod
    def from_csv(cls, csv_line: str, begin_time: datetime) -> 'Tick':
        values = csv_line.strip().split(',')
        open_price = float(values[1])
        max_price = float(values[2])
        min_price = float(values[3])
        close_price = float(values[4])
        volume = float(values[5])
        return cls(open_price, close_price, max_price, min_price, volume, begin_time)


class TickBar(Tick):
    def __init__(self, open_price: float, close_price: float, max_price: float, min_price: float, volume: float,
                 begin_time: datetime, end_time: datetime):
        super().__init__(open_price, close_price, max_price, min_price, volume, begin_time)
        self.end_time = end_time

    def in_period(self, timestamp: datetime) -> bool:
        return self.begin_time <= timestamp < self.end_time

    @classmethod
    def from_ticks(cls, ticks: Tuple[Tick]) -> 'TickBar':
        begin_time = ticks[0].begin_time
        end_time = ticks[-1].begin_time + ticks[-1].time_period
        open_price = ticks[0].open_price
        max_price = max(tick.max_price for tick in ticks)
        min_price = min(tick.min_price for tick in ticks)
        close_price = ticks[-1].close_price
        volume = sum(tick.volume for tick in ticks)
        return cls(open_price, close_price, max_price, min_price, volume, begin_time, end_time)

