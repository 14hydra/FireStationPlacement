#!/usr/bin/env bash
# Local heavy-experiment batch. Does NOT touch data/figures/final/01–10.
#
# Start (keeps Mac awake):
#   nohup caffeinate -dims bash scripts/run_local_heavy_experiments.sh \
#     >> data/figures/analysis/local_heavy_experiments/batch.log 2>&1 &
#   echo $! > data/figures/analysis/local_heavy_experiments/batch.pid
#
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

PY="${ROOT}/.venv/bin/python"
OUT="${ROOT}/data/figures/analysis/local_heavy_experiments"
mkdir -p "$OUT"
STATUS="${OUT}/status.json"
LOG_STEP="${OUT}/steps.log"
PIDFILE="${OUT}/batch.pid"
echo $$ > "$PIDFILE"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# Fresh status for this run
"$PY" - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
p = Path("${STATUS}")
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
p.write_text(json.dumps({
    "doc": "Local heavy-experiment batch (own computer).",
    "pid": $$,
    "started_at": now,
    "updated_at": now,
    "current": "batch",
    "steps": [{"step": "batch", "state": "running", "at": now, "detail": "started"}],
}, indent=2) + "\n")
PY

write_status() {
  local step="$1"
  local state="$2"
  local detail="${3:-}"
  STEP="$step" STATE="$state" DETAIL="$detail" STATUS_PATH="$STATUS" PID=$$ "$PY" - <<'PY'
import json, os
from pathlib import Path
from datetime import datetime, timezone
p = Path(os.environ["STATUS_PATH"])
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
data = json.loads(p.read_text())
steps = data.get("steps", [])
rec = {
    "step": os.environ["STEP"],
    "state": os.environ["STATE"],
    "at": now,
    "detail": os.environ.get("DETAIL", ""),
}
if steps and steps[-1].get("step") == rec["step"] and steps[-1].get("state") == "running":
    steps[-1] = rec
else:
    steps.append(rec)
data["steps"] = steps
data["updated_at"] = now
data["pid"] = int(os.environ["PID"])
data["current"] = rec["step"] if rec["state"] == "running" else None
p.write_text(json.dumps(data, indent=2) + "\n")
PY
}

run_step() {
  local name="$1"
  shift
  echo "" | tee -a "$LOG_STEP"
  echo "===== $(ts) START ${name} =====" | tee -a "$LOG_STEP"
  write_status "$name" "running"
  local t0=$SECONDS
  set +e
  "$@" >>"$LOG_STEP" 2>&1
  local rc=$?
  set +e
  local dt=$((SECONDS - t0))
  if [[ $rc -eq 0 ]]; then
    echo "===== $(ts) OK ${name} (${dt}s) =====" | tee -a "$LOG_STEP"
    write_status "$name" "ok" "${dt}s"
  else
    echo "===== $(ts) FAIL ${name} rc=${rc} (${dt}s) =====" | tee -a "$LOG_STEP"
    write_status "$name" "fail" "rc=${rc} after ${dt}s"
  fi
  return 0
}

echo "batch start $(ts) pid=$$" | tee -a "$LOG_STEP"

# PDF already scored; refresh quickly without refitting unless missing
if [[ ! -f data/figures/megacities/pdf_eval/summary.csv ]]; then
  run_step "megacity_pdf_evaluate" "$PY" scripts/megacity_pdf_evaluate.py --refit
else
  run_step "megacity_pdf_evaluate" "$PY" scripts/megacity_pdf_evaluate.py
fi

run_step "megacity_map_html" "$PY" scripts/build_megacity_map_html.py
run_step "placement_robustness" "$PY" scripts/placement_robustness.py
run_step "outer_error_check" "$PY" scripts/outer_error_check.py
run_step "nyc_fdny_validation" "$PY" scripts/validate_nyc_fdny_travel.py
run_step "neutral_siting" "$PY" scripts/neutral_siting_test.py
run_step "travel_time_variants" "$PY" scripts/compare_travel_time_variants.py
run_step "placement_tradeoff_busy" "$PY" scripts/placement_tradeoff.py --busy

"$PY" - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
out = Path("data/figures/analysis/local_heavy_experiments")
status = json.loads((out / "status.json").read_text())
steps = [s for s in status.get("steps", []) if s.get("step") != "batch"]
ok = sum(1 for s in steps if s.get("state") == "ok")
fail = sum(1 for s in steps if s.get("state") == "fail")
lines = [
    "# Local heavy experiments (own computer)",
    "",
    f"Finished (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
    "",
    f"- OK: {ok}",
    f"- Failed: {fail}",
    "",
    "| Step | State | Detail |",
    "|---|---|---|",
]
for s in steps:
    lines.append(f"| `{s.get('step')}` | {s.get('state')} | {s.get('detail','')} |")
lines += [
    "",
    "Does not modify `data/figures/final/01`–`10`.",
    "Log: `data/figures/analysis/local_heavy_experiments/steps.log`",
    "",
]
(out / "README.md").write_text("\n".join(lines) + "\n")
status["finished_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
status["current"] = None
status["summary"] = {"ok": ok, "fail": fail}
(out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
print("DONE", status["summary"])
PY

write_status "batch" "done"
echo "batch end $(ts)" | tee -a "$LOG_STEP"
rm -f "$PIDFILE"
