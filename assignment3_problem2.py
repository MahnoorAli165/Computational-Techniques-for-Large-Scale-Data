#!/usr/bin/env python3

from mrjob.job import MRJob
from mrjob.step import MRStep
import csv
import heapq

class MRTopKSystems(MRJob):

    def configure_args(self):
        super().configure_args()
        self.add_passthru_arg(
            '-k', type=int, default=10,
            help='Number of top systems to output'
        )

    def steps(self):
        return [
            MRStep(
                mapper=self.mapper_extract,
                reducer=self.reducer_sum
            ),
            MRStep(
                mapper=self.mapper_prepare_topk,
                reducer=self.reducer_topk
            )
        ]

   
    # STEP 1: Extract (system, RU) and sum values per system

    def mapper_extract(self, _, line):
        # Skip header
        if line.startswith("Constellation"):
            return

        row = next(csv.reader([line]))

        constellation = row[0].strip()
        star = row[1].strip()
        mineral_str = row[5].strip()

        # Determine system name
        if star == "Prime":
            system = constellation
        else:
            system = f"{star} {constellation}"

        # Convert mineral value
        try:
            mineral = int(mineral_str)
        except ValueError:
            return

        yield system, mineral

    def reducer_sum(self, system, values):
        yield system, sum(values)

    # STEP 2: Global top-k (single reducer)
    
    def mapper_prepare_topk(self, system, total):
        # Send all results to one reducer
        yield None, (system, total)

    def reducer_topk(self, _, system_total_pairs):
        k = self.options.k
        heap = []

        # Maintain a min-heap of size k
        for system, total in system_total_pairs:
            if len(heap) < k:
                heapq.heappush(heap, (total, system))
            else:
                heapq.heappushpop(heap, (total, system))

        # Sort descending before output
        topk_sorted = sorted(heap, reverse=True)

        for total, system in topk_sorted:
            yield system, total


if __name__ == '__main__':
    MRTopKSystems().run()
