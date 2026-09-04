#!/usr/bin/env bash
# SHIM (2026-09-04, R168 promotion): the 0.29 nvfp4 candidate IS the daily now — its launcher body moved to launch-daily.sh
# (EXP=1 / EXP=eval modes and every experiment knob kept). Existing experiment units (r168*/r169/r170/r172/r173*) call this
# path with EXP=1, so it forwards. New scripts should call launch-daily.sh directly.
exec bash /srv/qwen5090/launch-daily.sh "$@"
