#!/bin/bash
# synth_grid.py skips COMPLETE runs; re-invoke until each config has 50 (host OOM kills the process every ~13 runs)
cd ~/onto/v16/Windows
PY=~/onto/OntoSpan_Ubuntu_Experiments/.venv/bin/python
export SYN_S=0.3 SYN_Q=0.9
run() {  # K, LR_TYPE or empty, DIR
  for attempt in $(seq 1 30); do
    n=$(ls ~/onto/$3/*/COMPLETE.json 2>/dev/null | wc -l)
    [ "$n" -ge 50 ] && break
    if [ -n "$2" ]; then SYN_K=$1 SYN_LR_TYPE=$2 $PY synth_grid.py >> ~/onto/synth_main.log 2>&1
    else SYN_K=$1 $PY synth_grid.py >> ~/onto/synth_main.log 2>&1; fi
  done
}
run 10 "" synth_k10
run 10 0.001 synth_k10_typelr0.001
run 40 "" synth_k40
run 40 0.001 synth_k40_typelr0.001
