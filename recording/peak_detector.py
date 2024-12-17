from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


class PeakDetector:
    def __init__(self, array: list[float], lag: int, threshold: float, influence: float):
        self.y: list[float] = list(array)
        self.filtered_y = np.array(self.y).tolist()
        self.output = [0] * len(self.y)

        self.lag: int = lag
        self.threshold: float = threshold
        self.influence: float = influence

        self.recorded_region_start: Optional[int] = None
        self.recorded_regions: list[Tuple[int, int]] = []

        self.avg_filter: list[float] = [0] * len(self.y)
        self.std_filter: list[float] = [0] * len(self.y)

        if len(self.y) > self.lag:
            self.avg_filter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
            self.std_filter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()

    def mark_start_recording(self):
        self.recorded_region_start = len(self.y) - 1

    def mark_end_recording(self):
        if self.recorded_region_start is None:
            return

        self.recorded_regions.append((self.recorded_region_start, len(self.y) - 1))

    def plot(self):
        if self.influence != 1:
            plt.plot(self.y, label='unfiltered_y')
        plt.plot(self.filtered_y, label='y')
        plt.plot(self.avg_filter, label='avg_filter')
        plt.plot(self.std_filter, label='std_filter')

        for region in self.recorded_regions:
            plt.axvspan(region[0], region[1], color='g', alpha=0.1)

        plt.legend()
        plt.show()

    def add_value(self, new_value):
        self.y.append(new_value)

        i = len(self.y) - 1
        if i < self.lag:
            return 0
        elif i == self.lag:
            self.filtered_y = np.array(self.y).tolist()
            self.output = [0] * len(self.y)

            self.avg_filter: list[float] = [0] * len(self.y)
            self.avg_filter[self.lag] = np.mean(self.y[0:self.lag]).tolist()

            self.std_filter: list[float] = [0] * len(self.y)
            self.std_filter[self.lag] = np.std(self.y[0:self.lag]).tolist()
            return 0

        self.output += [0]
        self.filtered_y += [0]
        self.avg_filter += [0]
        self.std_filter += [0]

        if abs(self.y[i] - self.avg_filter[i - 1]) > (self.threshold * self.std_filter[i - 1]):

            if self.y[i] > self.avg_filter[i - 1]:
                self.output[i] = 1
            else:
                self.output[i] = -1

            self.filtered_y[i] = self.influence * self.y[i] + (1 - self.influence) * self.filtered_y[i - 1]
        else:
            if self.y[i] < self.avg_filter[i - 1]:
                self.output[i] = -1
            else:
                self.output[i] = 0

            self.filtered_y[i] = self.y[i]

        self.avg_filter[i] = np.mean(self.filtered_y[(i - self.lag):i])
        self.std_filter[i] = np.std(self.filtered_y[(i - self.lag):i])

        return self.output[i]
