#!/usr/bin/env python3

from mrjob.job import MRJob
import csv

class MRMineral(MRJob):

    def mapper(self, _, line):
        # Parse CSV line
        row = next(csv.reader([line]))

        # Skip header
        if row[0] == "Constellation":
            return

        constellation = row[0].strip()
        star = row[1].strip()
        mineral_str = row[5].strip()  # Mineral value (RU)

        # Build star system name (same logic as Problem 2)
        if star == "Prime":
            system_name = constellation
        else:
            system_name = f"{star} {constellation}"

        # Convert mineral value
        try:
            mineral_value = int(mineral_str)
        except ValueError:
            return

        yield system_name, mineral_value

    def reducer(self, system_name, mineral_values):
        yield system_name, sum(mineral_values)


if __name__ == '__main__':
    MRMineral().run()
