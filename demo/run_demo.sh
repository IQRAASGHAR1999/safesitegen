#!/usr/bin/env bash
# Live demonstration script. Each step pauses so you can talk over it.
# Usage:  bash demo/run_demo.sh
set -e
cd "$(dirname "$0")/.."

pause() { echo; read -rp "  [enter to continue] "; echo; }
rule()  { printf '\n\033[1m%s\033[0m\n%s\n' "$1" "$(printf '=%.0s' {1..72})"; }

rule "1. What the system knows"
python3 -m safesitegen classes
pause

rule "2. A trainer asks for a scenario in plain English"
python3 -m safesitegen prompt \
  "a storm drain crew working in an unshored trench with the spoil piled right on the edge" \
  --out demo/scene1
pause

rule "3. Open the training environment"
echo "  open demo/scene1/environment.html"
pause

rule "4. The trainer changes their mind"
python3 -m safesitegen prompt \
  "now add a crane working near the overhead power line and make it harder, but remove the spoil pile" \
  --modify demo/scene1/scenario.json --out demo/scene2
pause

rule "5a. What a system without a gate would have shipped"
python3 -m safesitegen prompt \
  "ironworker on a steel deck with an unprotected leading edge and an uncovered floor opening" \
  --fault-rate 0.8 --seed 8 --no-gate
pause

rule "5b. The same request, with the gate switched on"
python3 -m safesitegen prompt \
  "ironworker on a steel deck with an unprotected leading edge and an uncovered floor opening" \
  --fault-rate 0.8 --seed 8 --out demo/scene3
pause

rule "6. The same gate, measured at scale"
python3 experiments/run_ablation.py --n 120 --fault-rate 0.25 --out demo/results
