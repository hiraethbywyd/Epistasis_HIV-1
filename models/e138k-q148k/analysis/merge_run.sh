#!/usr/bin/env bash

# folder two levels above current directory
TOPDIR=$(basename "$(dirname "$(pwd)")")

# Process fluctuation files
cat r1/fluct.*.apf | grep -v '^#' | sort -k1,1n > r1fluct.dat
cat r1/fluct.*.apf | grep -v '^#' | sort -k1,1n > r2fluct.dat

paste r1fluct.dat r2fluct.dat \
  | awk '{print $1, ($2+$4)/2}' \
  > "${TOPDIR}_avg_fluct.dat"

rm r*fluct.dat

# Process dipole files
paste r1/avg_dipole.dat r2/avg_dipole.dat \
  | awk 'NR>1 {print $1, ($2+$4)/2}' \
  | sort -k1,1n \
  > "${TOPDIR}_avg_dipole.dat"

wait
mv ${TOPDIR}_*.dat ../../analysis/
