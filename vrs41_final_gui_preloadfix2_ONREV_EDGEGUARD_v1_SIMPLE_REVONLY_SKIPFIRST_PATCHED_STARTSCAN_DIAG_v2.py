MIN_COUNTED_STEPS = 0  # auto-patch
# ==== CALIBRATION (AUTO-PATCH) =====================================
# Use a single steps-per-nm; disable dir-aware kf/kb.
USE_DIR_AWARE_K = False
K_STEPS_PER_NM  = globals().get('K_STEPS_PER_NM', 361_765.0)
# Keep kf/kb if elsewhere referenced; they won't be used when USE_DIR_AWARE_K=False
kf = globals().get('kf', K_STEPS_PER_NM)
kb = globals().get('kb', K_STEPS_PER_NM)
# ==== CALIBRATION (AUTO-PATCH END) =================================


# Default slip magnitude (nm) for reversal-preload when no GUI override)
N_SLIP_DEFAULT_NM = 1.1618
import csv


# --- Defensive shim: ensure smart_preload_before_move is defined ---
try:
    smart_preload_before_move
except NameError:
    BACK_PRELOAD_ONLY = True
    TOL_BACK_PRELOAD_NM = 0.03
    def _get_optical_nm():
        try:
            return float(entry_current.get())
        except Exception:
            return float(state.get('current_nm', 0.0))
    def _get_motor_nm():
        try:
            anchor_nm = float(state.get('lambda_anchor_nm', _get_optical_nm()))
            anchor_pos = int(state.get('anchor_pos_steps', read_position()))
            cur_pos = int(read_position())
            spn = float(globals().get('STEPS_PER_NM', 361765))
            inv = bool(globals().get('INVERT_DIRECTION', True))
            delta_nm = (cur_pos - anchor_pos) / spn
            if inv: delta_nm = -delta_nm
            return anchor_nm + delta_nm
        except Exception:
            return float(state.get('current_nm', 0.0))


# [REMOVED 2025-08-29] Old maybe_preload_on_reversal policy removed to keep logic simple per user request.

from typing import Optional
import datetime
SUPPRESS_TIMEOUT_MESSAGES = True  # Hide [TIMEOUT]/timeout logs from the console
TIMEOUT_BASE = 3.0  # seconds (base)
TIMEOUT_PER_NM = 1.0  # seconds per nm (tune as needed)
TIMEOUT_MARGIN = 2.5  # safety multiplier

# ---- Slip compensation settings ----
# --- Portable settings path (relative to this script) ---
try:
    import os
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    SETTINGS_PATH = os.path.join(_THIS_DIR, "monochromator_settings.json")
except Exception:
    SETTINGS_PATH = "monochromator_settings.json"
# Base condition: consider preloading when moving to lower nm (backward scan)
PRELOAD_ON_BACKWARD = True
# Preload only when direction changes (or forced). Prevents cumulative drift.
PRELOAD_ONLY_ON_REVERSAL = True
# === Direction-reversal preload (2025-08-28) ===
# Only preload once when scan direction flips (forward↔backward).
# First leg (no previous direction) does NOT preload.
N_SLIP_NM = 1.1618               # preload magnitude in nm (opposite to new leg)
ALWAYS_ON_FIRST_LEG = False     # if True, also preload before very first leg
MIN_PRELOAD_WAIT_S = 0.20       # lower bound for wait during preload

def _compute_preload_wait_s(n_slip_nm: float) -> float:
    """Dynamic preload wait scaled by nm. Uses SEC_PER_NM if available."""
    try:
        sec_per_nm = float(globals().get('SEC_PER_NM', 10.0))
    except Exception:
        sec_per_nm = 10.0
    try:
        mn = float(globals().get('MIN_PRELOAD_WAIT_S', 0.20))
    except Exception:
        mn = 0.20
    return max(mn, abs(float(n_slip_nm)) * sec_per_nm)

def maybe_preload_on_reversal(next_dir: int, preload_nm: float = None) -> bool:
    """Preload only when direction flips. Returns True iff a preload was issued."""
    # Guard config
    try:
        only_on_rev = bool(globals().get('PRELOAD_ONLY_ON_REVERSAL', True))
    except Exception:
        only_on_rev = True
    if not only_on_rev:
        return False

    # Normalize direction
    try:
        nd = int(next_dir)
        if nd == 0:
            return False
    except Exception:
        nd = -1 if str(next_dir).strip() == '-1' else 1

    # Determine previous direction
    try:
        last = int(state.get('last_nm_dir', 0) or 0)
    except Exception:
        last = 0

    # First leg handling
    if last == 0 and not ALWAYS_ON_FIRST_LEG:
        # No previous direction -> skip preload
        return False
    # No reversal
    if last == nd:
        return False

    # Reversal detected => preload opposite to new direction

    try:

        n_slip_nm = float(preload_nm) if preload_nm is not None else float(state.get('n_slip_nm_back', globals().get('N_SLIP_DEFAULT_NM', 1.1618)))

    except Exception:

        n_slip_nm = float(globals().get('N_SLIP_DEFAULT_NM', 1.1618))
    preload_dir = -nd  # opposite to new leg
    try:
        steps = steps_for_delta_nm(n_slip_nm * preload_dir)
    except Exception:
        # Fallback if helper is missing
        spn = float(globals().get('STEPS_PER_NM', 361765.0))
        inv = bool(globals().get('INVERT_DIRECTION', True))
        raw_steps = int(round(n_slip_nm * spn * (preload_dir)))
        steps = -raw_steps if inv else raw_steps

    try:
        log(f"[REV-PRELOAD] dir flip {last:+d}->{nd:+d}; issuing LR{steps} (~{preload_dir*n_slip_nm:+.3f} nm)")
    except Exception:
        pass

    
    # --- Edge guard: avoid crossing software limits with the preload ---
    try:
        base_motor_nm = float(state.get('motor_lambda_nm', state.get('current_nm', 0.0)))
        target_preload_nm = base_motor_nm + (preload_dir * n_slip_nm)
        try:
            ok_limits = check_soft_limits_nm(target_preload_nm)
        except Exception:
            ok_limits = True  # if guard unavailable, allow
        if not ok_limits:
            log(f"[REV-PRELOAD] skipped: would cross soft limit at target {target_preload_nm:.3f} nm", "warn")
            return False
    except Exception as _e_guard:
        # If anything odd, proceed but log
        try:
            log(f"[REV-PRELOAD] edge-guard check failed: {_e_guard}. Proceeding without guard.", "warn")
        except Exception:
            pass
# Clean start: recoverable stop + drain
    try:
        routine_stop_pause()
    except Exception:
        # fall back to minimal quiet stop
        try:
            send_quiet("ST"); send_quiet("V0")
        except Exception:
            pass
    try:
        drain_rx_quiet()
    except Exception:
        pass

    # Send preload move
    try:
        # Compute target before sending, based on current position
        cur_abs = read_position()
    except Exception:
        try:
            cur_abs = read_position_retry()
        except Exception:
            cur_abs = None
    try:
        send_quiet(f"LR{int(steps)}"); send_quiet("M")
    except Exception:
        return False

    # Wait for preload completion (prefer position with dynamic timeout)
    import time
    deadline = _compute_preload_wait_s(n_slip_nm)
    ok = False
    try:
        if cur_abs is not None:
            target_abs = cur_abs + int(steps)
            ok = wait_until_position(target_abs, timeout=deadline*1.5)
        else:
            ok = False
    except Exception:
        ok = False
    if not ok:
        # Fallback: sleep by dynamic time
        try:
            time.sleep(deadline)
        except Exception:
            pass

    # Dynamic settle: use PRELOAD_SETTLE_S if fixed, else compute from move time
    try:
        settle_fixed = globals().get('PRELOAD_SETTLE_S', None)
        if settle_fixed is None:
            sec_per_nm = float(globals().get('SEC_PER_NM', 10.0))
            frac = float(globals().get('PRELOAD_SETTLE_FRAC_OF_MOVE', 0.10))
            base = float(globals().get('PRELOAD_SETTLE_BASE_S', 0.10))
            min_settle = float(globals().get('PRELOAD_SETTLE_MIN_S', 0.20))
            settle_s = max(min_settle, base + frac * (abs(n_slip_nm) * sec_per_nm))
        else:
            settle_s = float(settle_fixed)
        if settle_s > 0:
            time.sleep(settle_s)
    except Exception:
        pass

    # Update motor lambda only; optical stays unchanged
    try:
        motor_nm_old = float(state.get('motor_lambda_nm', state.get('current_nm', 0.0)))
        motor_nm_new = motor_nm_old + (preload_dir * n_slip_nm)
        state['motor_lambda_nm'] = motor_nm_new
        try:
            # Optional GUI hook
            if 'gui' in globals() and hasattr(gui, 'update_motor_lambda'):
                gui.update_motor_lambda(motor_nm_new)
        except Exception:
            pass
        try:
            log(f"[REV-PRELOAD] complete; motorλ {motor_nm_old:.6f}→{motor_nm_new:.6f} nm; opticalλ unchanged")
        except Exception:
            pass
    except Exception:
        pass

    return True

# Force preload on the first backward move in a sequence (e.g., scan start or after STOP)
PRELOAD_ON_FIRST_BACKWARD = True

# --- Dynamic preload settle (speed-aware) ---
# Set PRELOAD_SETTLE_S=None to compute: settle = max(MIN, BASE + FRAC * (nm_slip * SEC_PER_NM))
SEC_PER_NM = 10.0  # seconds per nm (tune if motor speed changes)
PRELOAD_SETTLE_FRAC_OF_MOVE = 0.10
PRELOAD_SETTLE_MIN_S = 0.20
PRELOAD_SETTLE_MAX_S = 0.30
PRELOAD_SETTLE_BASE_S = 0.10
PRELOAD_SETTLE_S = None  # None => dynamic; set a fixed seconds value to force constant settle

from tkinter import filedialog as fd
from tkinter import ttk
from tkinter import scrolledtext



def timeout_for_target_steps(target_steps: int, fallback_nm) -> float:
    """
    Compute timeout based on *remaining* distance to target from current position.
    If anything fails, fall back to the provided nm estimate.
    """
    try:
        pos_now = read_position()
        dist_nm = abs((target_steps - pos_now) / float(STEPS_PER_NM))
        return timeout_for_nm(dist_nm)
    except Exception:
        try:
            return timeout_for_nm(abs(float(fallback_nm)))
        except Exception:
            return timeout_for_nm(0.0)


def timeout_for_nm(delta_nm: float) -> float:
    try:
        dist = abs(float(delta_nm))
    except Exception:
        return max(MIN_MOVE_TIMEOUT, TIMEOUT_OVERHEAD_S + 1.0)
    try:
        _dwell = float(entry_wait.get())
    except Exception:
        _dwell = 0.0
    settle_overhead = min(5.0, 0.25 * _dwell)
    t = SEC_PER_NM * dist + TIMEOUT_OVERHEAD_S + settle_overhead
    if t < MIN_MOVE_TIMEOUT: t = MIN_MOVE_TIMEOUT
    if t > MAX_MOVE_TIMEOUT: t = MAX_MOVE_TIMEOUT
    return t

# LAB TRY VARIANT: sets INVERT_DIRECTION=True so increasing wavelength sends negative LR steps.
# NOTE: this version fixes a broken import line ("import" on one line and modules on the next).
# All other logic unchanged from your PATCHED version.
# vers30_labview_dualcheck_FINALREADY_quickpatch3.py
# Monochromator control – Ready-to-run (GUI + FSM) using OST/POS (no SDO)
# - COM-stable LabVIEW-style init/stop (EN/HP0/V0, ST/HP0/V0)
# - Motion completion by position (POS), not bit10
# - Go To wavelength + Scan (pause/resume/stop)
# - Clean console output, no CSV logging
# - Optional DAQ read (auto-disabled if nidaqmx not installed)

import sys, os, time, threading, queue, random
import time
import serial
import re


# --- Passive post-move Statusword (0x6041) audit (non-blocking) ---
USE_POSTMOVE_STATUSWORD_AUDIT = False  # set False to disable audit
STATUS_FAULT_BIT = 1 << 3
STATUS_TR_BIT = 1 << 10
STATUS_INTERNAL_LIMIT_BIT = 1 << 11  # if supported by your drive

def postmove_status_audit(where: str = "") -> bool:
    """
    Read 0x6041 once after a *POS-confirmed* move.
    If FAULT or INTERNAL_LIMIT is set, stop scanning and return False.
    Never blocks motion timing; POS remains the source of truth.
    """
    if not USE_POSTMOVE_STATUSWORD_AUDIT:
        return True
    try:
        sw = _read_statusword()  # your existing SDO read; if it raises, we just continue
    except Exception as e:
        try:
            log(f"[AUDIT] 0x6041 read skipped: {e}")
        except Exception:
            pass
        return True

    bad = []
    if (sw & STATUS_FAULT_BIT):          bad.append("FAULT")
    if (sw & STATUS_INTERNAL_LIMIT_BIT): bad.append("INTERNAL_LIMIT")

    if bad:
        try:
            log(f"[ERROR] Post-move audit{(' ['+where+']') if where else ''}: {'+'.join(bad)} (SW=0x{sw:04X}). Stopping.", "error")
        except Exception:
            pass
        try:
            state["stop_flag"] = True
        except Exception:
            pass
        return False

    # Informational only — do NOT gate motion on TR
    if not (sw & STATUS_TR_BIT):
        try:
            log(f"[WARN] Post-move audit{(' ['+where+']') if where else ''}: TR=0 (SW=0x{sw:04X}) — POS ok; continuing.", "warn")
        except Exception:
            pass

    return True


# --- Input validation (LabVIEW-like) ---
def _float_or_none(val):
    try:
        return float(val)
    except Exception:
        return None

def validate_goto_input():
    try:
        tgt = _float_or_none(entry_goto.get())
    except Exception:
        tgt = None
    if tgt is None:
        log("[ERROR] GoTo: invalid target wavelength.", "error")
        return None
    # Optional: software limits
    try:
        min_nm = float(state.get("soft_min_nm", "-inf"))
        max_nm = float(state.get("soft_max_nm", "inf"))
    except Exception:
        min_nm, max_nm = -float("inf"), float("inf")
    if not (min_nm <= tgt <= max_nm):
        log(f"[ERROR] GoTo: target {tgt} nm outside allowed range [{min_nm}, {max_nm}] nm.", "error")
        return None
    return tgt

def validate_scan_inputs():
    # Collect entry values
    try:
        s = _float_or_none(entry_start.get())
        e = _float_or_none(entry_end.get())
        stp = _float_or_none(entry_step.get())
        dwell = _float_or_none(entry_wait.get())
    except Exception:
        s = e = stp = dwell = None

    # Basic checks
    if s is None or e is None or stp is None:
        log("[ERROR] Scan: start/end/step must be numeric.", "error")
        return None
    if stp <= 0:
        log("[ERROR] Scan: step must be > 0.", "error")
        return None
    # Allow descending scans (end < start); direction handled later.
    if dwell is None or dwell < 0:
        log("[WARN] Scan: dwell invalid; using 0 s.", "warn")
        dwell = 0.0

    # Optional: software limits
    try:
        min_nm = float(state.get("soft_min_nm", "-inf"))
        max_nm = float(state.get("soft_max_nm", "inf"))
    except Exception:
        min_nm, max_nm = -float("inf"), float("inf")
    if not (min_nm <= s <= max_nm) or not (min_nm <= e <= max_nm):
        log(f"[ERROR] Scan: range [{s}, {e}] nm outside allowed [{min_nm}, {max_nm}] nm.", "error")
        return None

    return {"start": s, "end": e, "step": stp, "dwell": dwell}
import tkinter as tk

# ---- Limits-only Statusword audit (no TR use, non-blocking) ----
USE_POSTMOVE_LIMIT_AUDIT = True  # set False to disable
STATUS_FAULT_BIT = 1 << 3
STATUS_TR_BIT = 1 << 10
STATUS_INTERNAL_LIMIT_BIT = 1 << 11  # "internal limit active" per CiA-402 / Faulhaber docs
STATUS_WARNING_BIT = 1 << 7          # optional (warn only)
# ---- OST (Operational Status) masks (fallback when SDO 0x6041 is unavailable) ----
OST_FAULT_MASK = 0x0000  # placeholder; updated after OST_ERROR_MASK is defined
OST_HARD_LIMIT_MASK = 0x0000  # legacy disabled     # example: hard/limit reached (256)
OST_INTERNAL_LIMIT_MASK = 0x0000  # legacy disabled # example: internal software limit (1024)



def postmove_limit_audit(where: str = "") -> bool:
    """
    Use OST-only to detect FAULT or LIMIT conditions after a motion step.
    - Returns True if OK (no limit/fault), False if a limit/fault occurred.
    - Does NOT use bit-10 or any 0x6041 SDO reads.
    """
    if not USE_POSTMOVE_LIMIT_AUDIT:
        return True
    ost = read_ost()
    if ost is None:
        try:
            log("[AUDIT] OST read skipped: no response.")
        except Exception:
            pass
        return True  # don't fail the move if we can't audit
    if ost & (OST_FAULT_MASK | OST_HARD_LIMIT_MASK | OST_INTERNAL_LIMIT_MASK):
        try:
            log(f"[ERROR] Post-move limit/fault via OST{(' ['+where+']') if where else ''}: 0x{ost:04X}.", "error")
        except Exception:
            pass
        return False
    try:
        log(f"[AUDIT] Post-move OST OK: 0x{ost:04X}{(' ['+where+']') if where else ''}.")
    except Exception:
        pass
    return True

def _queue_finished_prompt(*args, **kwargs):
    # Disabled at import-time; real prompt is scheduled after the queue ends.
    return False

def _maybe_prompt_save_all_scans_after_queue():
    """Show the 'Queue finished' save prompt only after the queue has actually finished."""
    try:
        have_scans = bool(graph_data.get("scans")) and len(graph_data.get("scans") or []) > 0
    except Exception:
        have_scans = False
    if not have_scans:
        return
    try:
        from tkinter import messagebox
        if messagebox.askyesno("Queue finished", "Save all scans and queue summary?"):
            try:
                export_all_scans_and_summary()  # existing function in your file
            except Exception as e:
                log(f"[ERROR] Export failed: {e}", "error")
    except Exception as e:
        try:
            log(f"[WARN] Could not show save prompt: {e}", "warn")
        except Exception:
            pass
# --- END PATCH ---


# ---------- Device & UI constants ----------
# IMPORTANT: steps-per-nm calibration. Adjust if your rig differs.

# --- Calibration persistence helpers (non-destructive JSON update) ---
def _read_settings_json():
    try:
        import json, os
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _write_settings_json(data: dict):
    try:
        import json, os
        os.makedirs(os.path.dirname(SETTINGS_PATH) or ".", exist_ok=True)
        with open(SETTINGS_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        try:
            log(f"[SETTINGS] Save failed: {e}", "warn")
        except Exception:
            pass
        return False

def _load_steps_per_nm_default():
    d = _read_settings_json()
    try:
        return int(d.get('steps_per_nm', 361765))
    except Exception:
        return 361765

def _save_steps_per_nm(val: int):
    d = _read_settings_json()
    d['steps_per_nm'] = int(val)
    _write_settings_json(d)
    try:
        log(f"[CAL] Saved steps_per_nm={int(val)}")
    except Exception:
        pass

# merged writer for n_slip (in steps)
def _save_n_slip(val: int):
    d = _read_settings_json()
    d['n_slip_steps'] = int(val)
    _write_settings_json(d)
    try:
        log(f"[SLIP] Saved n_slip={int(val)} steps")
    except Exception:
        pass

STEPS_PER_NM = _load_steps_per_nm_default()
SEC_PER_NM = globals().get('SEC_PER_NM', 10.0)
TIMEOUT_OVERHEAD_S = globals().get('TIMEOUT_OVERHEAD_S', 5.0)
BASE_TIMEOUT_MIN = globals().get('BASE_TIMEOUT_MIN', 8.0)


DEFAULT_PORT = "COM3"
DEFAULT_BAUD = 9600
POLL_INTERVAL = 0.05                 # s for motion polling

# ---- Statusword (CiA-402) integration toggles ----
USE_SDO_FOR_AUDIT = False  # Disable SDO/0x6041 audits; use OST only

USE_STATUSWORD = False        # keep False until you verify in lab
STATUSWORD_NODE = 1           # typical node ID; adjust if needed
STATUS_ACK_ARM_TIMEOUT_S = 0.5  # time to see bit-12 ACK / bit-10 drop
STATUS_TR_DEBOUNCE_POLLS = 2    # consecutive polls TR must stay high
STATUS_POLL_S = 0.05            # 50 ms, in LabVIEW’s typical range

# Statusword bits (CiA-402 0x6041)
STATUS_FAULT_BIT = 1 << 3   # Fault
STATUS_TR_BIT    = 1 << 10  # Target Reached
# --- Soft limits + post-move Statusword audit (passive; non-blocking for motion) ---
# Configure these if you want software limits; leave as None to disable soft limits.
POST_MOVE_STATUS_AUDIT = True  # Read 0x6041 once after a successful move/step and log faults/limits
SOFT_LIMIT_MIN_NM = globals().get('SOFT_LIMIT_MIN_NM', None)
SOFT_LIMIT_MAX_NM = globals().get('SOFT_LIMIT_MAX_NM', None)
SOFT_LIMIT_MARGIN_NM = float(globals().get('SOFT_LIMIT_MARGIN_NM', 0.0))

def _get_bit_default(name, default):
    try:
        return int(globals().get(name, default))
    except Exception:
        return default

STATUS_FAULT_BIT = _get_bit_default('STATUS_FAULT_BIT', 1 << 3)   # Fault
STATUS_TR_BIT    = _get_bit_default('STATUS_TR_BIT',    1 << 10)  # Target Reached
STATUS_INT_LIMIT_BIT = _get_bit_default('STATUS_INT_LIMIT_BIT', 1 << 11)  # Internal limit active (if supported)

def check_soft_limits_nm(target_nm: float) -> bool:
    """Return True if target is inside optional software limits (with margin)."""
    try:
        mn = SOFT_LIMIT_MIN_NM
        mx = SOFT_LIMIT_MAX_NM
        margin = float(SOFT_LIMIT_MARGIN_NM)
        if mn is not None and target_nm < float(mn) + margin:
            log(f"[ERROR] Target λ {target_nm:.3f} nm < min limit {float(mn)+margin:.3f} nm. Move aborted.", "error")
            return False
        if mx is not None and target_nm > float(mx) - margin:
            log(f"[ERROR] Target λ {target_nm:.3f} nm > max limit {float(mx)-margin:.3f} nm. Move aborted.", "error")
            return False
    except Exception:
        pass
    return True

def post_move_statusword_audit(tag: str = "post-move") -> bool:
    """Passive 0x6041 audit after POS-confirmed success.
    Never used to decide success of the move itself; only logs and sets stop flag if severe bits are set.
    """
    if not POST_MOVE_STATUS_AUDIT:
        return True
    try:
        sw = _read_statusword()
    except Exception:
        # SDO not available is not an error; skip audit quietly
        log("[INFO] Statusword audit skipped (SDO unavailable).", "info")
        return True

    faults = []
    if sw & STATUS_FAULT_BIT: faults.append("FAULT")
    if sw & STATUS_INT_LIMIT_BIT: faults.append("INTERNAL_LIMIT")

    if faults:
        log(f"[ERROR] {tag}: drive reports {'+'.join(faults)}. Stopping queue/scan.", "error")
        try:
            state["stop_flag"] = True
        except Exception:
            pass
        return False
    return True

STATUS_ACK_BIT   = 1 << 12  # Set-Point Acknowledge

# -------------------------------------------

# Optional plotting (works without DAQ)
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------- Plot Manager (XY Graph with multi-scan + jog) --------
class PlotManager:
    """
    Minimal, self-contained plotting helper.
    - Maintains multiple scan series (lines) and one jog layer (markers only).
    - Decimates redraws to ~15 Hz using root.after; thread-safe appends.
    - No effect on FSM/serial; only called from existing hooks.
    """
    def __init__(self, root, ax, canvas):
        import threading
        self.root = root
        self.ax = ax
        self.canvas = canvas
        self._lock = threading.Lock()
        self._plots = {}           # plot_id -> {"x": [], "y": [], "line": Line2D, "label": str, "status": str}
        self._active_id = None
        # Dedicated jog layer (markers only)
        (jog_line,) = ax.plot([], [], linestyle="None", marker="o", label="Jog")
        self._jog = {"x": [], "y": [], "line": jog_line}
        # Axes labels (keep existing style if already set elsewhere)
        if not ax.get_xlabel(): ax.set_xlabel("Wavelength (nm)")
        if not ax.get_ylabel(): ax.set_ylabel("Signal (V)")
        # Start redraw loop
        self._schedule_redraw()

    def _schedule_redraw(self):
        self.root.after(66, self._redraw)  # ~15 Hz

    def _redraw(self):
        with self._lock:
            # Update all series
            for meta in self._plots.values():
                meta["line"].set_data(meta["x"], meta["y"])
            self._jog["line"].set_data(self._jog["x"], self._jog["y"])
            # Auto-scale smartly
            self.ax.relim()
            self.ax.autoscale_view()
            # Legend: show one entry per scan + Jog
            try:
                self.ax.legend(loc="best", frameon=False)
            except Exception:
                pass
            self.canvas.draw_idle()
        self._schedule_redraw()

    def start_new_scan(self, label: str, meta: dict) -> str:
        # Create a new line with automatic color
        (line,) = self.ax.plot([], [], label=label)
        pid = f"scan_{len(self._plots)+1}"
        with self._lock:
            self._plots[pid] = {"x": [], "y": [], "line": line, "label": label, "status": "active", "meta": dict(meta or {})}
            self._active_id = pid
        return pid

    def set_active(self, plot_id: str):
        with self._lock:
            if plot_id in self._plots:
                self._active_id = plot_id

    def append_scan_point(self, plot_id: str, x_nm: float, y_val: float):
        with self._lock:
            meta = self._plots.get(plot_id)
            if not meta or meta["status"] != "active":
                return
            meta["x"].append(float(x_nm))
            meta["y"].append(float(y_val))

    def finalize_scan(self, plot_id: str, status: str):
        with self._lock:
            meta = self._plots.get(plot_id)
            if meta:
                meta["status"] = status

    def append_jog_point(self, x_nm: float, y_val: float):
        with self._lock:
            self._jog["x"].append(float(x_nm))
            self._jog["y"].append(float(y_val))

    def clear_all(self):
        with self._lock:
            # Remove lines from axes
            for meta in self._plots.values():
                try:
                    meta["line"].remove()
                except Exception:
                    pass
            try:
                self._jog["line"].remove()
            except Exception:
                pass
            self._plots.clear()
            # Recreate jog layer
            (jog_line,) = self.ax.plot([], [], linestyle="None", marker="o", label="Jog")
            self._jog = {"x": [], "y": [], "line": jog_line}
            self._active_id = None
        # Force a redraw
        self.canvas.draw_idle()

# -------- Configuration --------


# --- Hardened Serial I/O (minimal, high-impact) ---
# Keep the link stable without slowing down the UI or scan loops.
ECHO_FILTER = False          # Set True if the controller echoes the command back
RESYNC_ON_TIMEOUT = True     # On timeout: drain buffers and probe with a status cmd (POS)
WRITE_SETTLE_S = 0.01        # Short post-flush delay so the device can breathe
CMD_TIMEOUT = 1.0            # Standard command timeout (s)
POLL_TIMEOUT = 0.10          # Short read timeout used in polling loops so Stop/Pause stay responsive
# --------------------------------------------------

# --- Timeouts ---
MOVE_TIMEOUT = 60.0                 # s, fallback cap for a single move
# Dynamic move timeout model: 1 nm ≈ 10 s + overhead (clamped)
MIN_MOVE_TIMEOUT = 10.0
MAX_MOVE_TIMEOUT = 900.0
# -----------------

POS_TOL_STEPS = 1000                    # steps tolerance to consider reached
# Apply generous floor to avoid spurious timeouts
POS_TOL_STEPS = max(int(0.01 * STEPS_PER_NM), int(POS_TOL_STEPS)); POS_TOL_STEPS_FLOOR_APPLIED = True


try:
    POS_TOL_STEPS_RESUME
except NameError:
    try:
        POS_TOL_STEPS_RESUME = max(POS_TOL_STEPS, int(float(STEPS_PER_NM) * 0.001))
    except Exception:
        POS_TOL_STEPS_RESUME = int(float(STEPS_PER_NM) * 0.001)


POS_STABLE_COUNT = 5                 # consecutive hits to confirm reached
DEFAULT_BAUD = 9600                  # your controller's baud
DEFAULT_PORT = "COM3"                # change in GUI
INVERT_DIRECTION = True             # set True if nm+ should send negative LR  # LAB TRY: forward λ uses negative LR on this rig
SAFE_AFTER_MOVE = True               # send V0 after each move
# --------------------------------

# Optional DAQ (auto-disable if not available)
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    DAQ_AVAILABLE = True
except Exception:
    nidaqmx = None
    AcquisitionType = None
    TerminalConfiguration = None
    DAQ_AVAILABLE = False


def _load_n_slip_default():
    """Load n_slip (steps) from JSON; default to 1.618 nm if missing."""
    try:
        import json, os
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, 'r') as f:
                data = json.load(f)
            return int(data.get('n_slip_steps', int(round(1.618*STEPS_PER_NM))))
    except Exception:
        pass
    return int(round(1.618*STEPS_PER_NM))

def _save_n_slip(val: int):
    try:
        import json
        with open(SETTINGS_PATH, 'w') as f:
            json.dump({'n_slip_steps': int(val)}, f, indent=2)
        log(f"[SLIP] Saved n_slip={int(val)} steps")
    except Exception as e:
        log(f"[SLIP] Could not save n_slip: {e}", 'warn')

state = {
    
    'n_slip_steps': _load_n_slip_default(),
    'last_nm_dir': 0,
    'force_preload_next': True,
    'last_preload_ts': None,
    'drive_enabled': False,
'limit_blocked': False,
    "scan_queue": [],

    "ser": None,
    "connected": False,
    "is_scanning": False,
    "is_paused": False,
    "stop_flag": False,
    "current_nm": 500.0,         # user-tracked; device doesn't report nm
    "fsm": "IDLE",
    "paused_mid_move": False,
    "pause_requested": False,
    "current_step_target_abs": None,
}




# Mirror generic n_slip to backward-specific key for preload logic
try:
    state["n_slip_steps_back"] = int(state.get("n_slip_steps", 0))
except Exception:
    state["n_slip_steps_back"] = 0
# Default nm slip fallback for GUI/editable use
try:
    state.setdefault('n_slip_nm_back', float(globals().get('N_SLIP_DEFAULT_NM', 1.1618)))
except Exception:
    pass

# thread-safe writes to serial
state['lock'] = threading.Lock()
# ------------------ FSM Helpers ------------------
FSM_STATES = {"IDLE","MOVING","SCAN","PAUSED","ERROR"}

def set_fsm(next_state: str):
    if next_state not in FSM_STATES:
        log(f"[FSM] Invalid state: {next_state}", "warn")
        return
    prev = state.get("fsm","IDLE")
    state["fsm"] = next_state
    log(f"[FSM] {prev} -> {next_state}")


# ------------------ Low-level serial helpers ------------------
def ser_ok():
    return state["ser"] is not None and state["ser"].is_open

def send_quiet(cmd: str):
    """Write-only command (no reply expected). For EN/V0/DI/LR/M etc."""
    if not ser_ok():
        return
    if not cmd.endswith("\r"):
        cmd += "\r"
    with state["lock"]:
        s = state["ser"]
        try:
            s.reset_output_buffer()
            s.write(cmd.encode("ascii"))
            s.flush()
        except Exception:
            pass

# ---- RX drain + stop helpers (injected) ----
def drain_rx_quiet(max_wait_s: float = 1.5, quiet_window_s: float = 0.25):
    """Read and discard controller output until input stays quiet."""
    try:
        s = state.get("ser")
        if not s: 
            return
        t0 = time.time()
        last_rx = t0
        try:
            s.timeout = max(s.timeout or 0.2, 0.2)
        except Exception:
            pass
        while True:
            try:
                n = s.in_waiting
            except Exception:
                n = 0
            if n and n > 0:
                try:
                    _ = s.read(n)
                    last_rx = time.time()
                except Exception:
                    break
            if (time.time() - last_rx) >= quiet_window_s:
                break
            if (time.time() - t0) >= max_wait_s:
                break
    except Exception:
        pass

def routine_stop_pause():
    """Recoverable stop: ST -> (HP0) -> V0 -> drain RX -> optional statusword audit."""
    try:
        send_quiet("ST")
        try:
            send_quiet("HP0")
        except Exception:
            pass
        send_quiet("V0")
        drain_rx_quiet()
        if USE_STATUSWORD:
            try:
                sw = _read_statusword()
                log(f"[STOP] Routine stop; statusword=0x{sw:04X}")
            except Exception:
                log("[STOP] Routine stop; statusword read skipped.")
        # Do NOT DI here; keep drive enabled
        state["drive_enabled"] = True
        update_drive_status()
        set_fsm("IDLE")
    except Exception as e:
        log(f"[STOP] Routine stop issue: {e}", "warn")

def hard_kill_and_reinit():
    """Hard kill + reinitialize: ST -> DI -> small delay -> EN -> V0 -> init_motor()."""
    try:
        send_quiet("ST")
        send_quiet("DI")
        time.sleep(0.15)
        drain_rx_quiet()
        send_quiet("EN")
        send_quiet("V0")
        # Mark as disabled until init confirms
        state["drive_enabled"] = False
        update_drive_status()
        init_motor()
        set_fsm("IDLE")
        log("[STOP] Hard kill & reinit complete.")
    except Exception as e:
        log(f"[STOP] Hard kill failed: {e}", "error")



def send_cmd(cmd: str, timeout: float = 1.0) -> str:
    """Send ASCII cmd with CR and read CR-terminated reply. Clean, quiet."""
    if not ser_ok():
        return ""
    if not cmd.endswith("\r"):
        cmd += "\r"
    with state["lock"]:
        s = state["ser"]
        try:
            s.reset_input_buffer()
            s.reset_output_buffer()
            s.write(cmd.encode("ascii"))
            s.flush()
            s.timeout = timeout
            raw = s.read_until(b"\r").decode(errors="ignore").strip()
            # RESYNC on timeout/empty response (optional)
            if (not raw) and RESYNC_ON_TIMEOUT:
                try:
                    log(f"[RESYNC] Timed out on {cmd.strip()!r}; draining and probing POS.")
                except Exception:
                    pass
                try:
                    s.reset_input_buffer(); s.reset_output_buffer()
                    time.sleep(max(0.0, float(WRITE_SETTLE_S)))
                    s.write(b"POS\r"); s.flush()
                    s.timeout = max(0.2, float(POLL_TIMEOUT)) if "POLL_TIMEOUT" in globals() else 0.5
                    _ = s.read_until(b"\r")  # ignore content
                except Exception:
                    pass
            return raw
        except Exception as e:
            log(f"[SERIAL ERROR] {e}", "error")
            return ""


# ---- OST verbose logging controls ----
LOG_OST_VERBOSE = True           # set False to silence periodic OST logs
OST_LOG_THROTTLE_S = 0.5         # minimum seconds between logs
# These live in state so we persist between calls
state.setdefault("last_ost_log_ts", 0.0)
state.setdefault("last_ost_value", None)

# ==== FAULHABER OST bit meanings (RS232 manual, Table 8) ====
OST_HOMING_RUNNING       = 0x0001  # bit0
OST_PROG_RUNNING         = 0x0002  # bit1
OST_STOPPED_DELAY        = 0x0004  # bit2
OST_STOPPED_NOTIFY       = 0x0008  # bit3 (program stop notify; not a fault)
OST_CURRENT_LIMIT_ACTIVE = 0x0010  # bit4
OST_DEVIATION_ERROR      = 0x0020  # bit5
OST_OVERVOLTAGE          = 0x0040  # bit6
OST_OVERTEMPERATURE      = 0x0080  # bit7
# Status inputs (digital inputs)
OST_STATUS_INPUT1        = 0x0100  # bit8
OST_STATUS_INPUT2        = 0x0200  # bit9
OST_STATUS_INPUT3        = 0x0400  # bit10
# Outputs reserved 13..15
OST_POSITION_ATTAINED    = 0x10000 # bit16
OST_LIMIT_CONT_CURRENT   = 0x20000 # bit17
# Masks used by app
OST_ERROR_MASK = (OST_DEVIATION_ERROR | OST_OVERVOLTAGE | OST_OVERTEMPERATURE)
OST_FAULT_MASK = OST_ERROR_MASK  # legacy alias now resolved
OST_INPUT_MASK = (OST_STATUS_INPUT1 | OST_STATUS_INPUT2 | OST_STATUS_INPUT3)

def _decode_ost_flags(ost: int) -> str:
    parts = []
    if ost & OST_POSITION_ATTAINED: parts.append("POS_ATTAINED")
    if ost & OST_CURRENT_LIMIT_ACTIVE: parts.append("CUR_LIMIT")
    if ost & OST_DEVIATION_ERROR: parts.append("DEVIATION")
    if ost & OST_OVERVOLTAGE: parts.append("OVERVOLT")
    if ost & OST_OVERTEMPERATURE: parts.append("OVERTEMP")
    if ost & OST_STATUS_INPUT1: parts.append("IN1")
    if ost & OST_STATUS_INPUT2: parts.append("IN2")
    if ost & OST_STATUS_INPUT3: parts.append("IN3")
    return "|".join(parts) if parts else "—"
def read_ost(timeout: float = 0.5):
    """
    Read FAULHABER Operational Status (OST) via ASCII "OST" command.
    Returns an int or None if no response/parse error.
    """
    try:
        raw = send_cmd("OST", timeout=timeout) or ""
        # Extract first integer in the reply (decimal or hex)
        import re as _re
        m = _re.search(r"(0x[0-9A-Fa-f]+|\d+)", raw)
        if not m:
            return None
        token = m.group(1)
        val = int(token, 16) if token.lower().startswith("0x") else int(token)
        # Throttled OST log with decoded flags
        try:
            import time as _t
            now_ts = _t.time()
            last_ts = state.get("last_ost_log_ts", 0.0)
            last_val = state.get("last_ost_val", None)
            if (now_ts - last_ts) >= 0.5 or (last_val is None or int(last_val) != int(val)):
                try:
                    log(f"[OST] 0x{val:04X} ({_decode_ost_flags(val)})")
                except Exception:
                    pass
                state["last_ost_log_ts"] = now_ts
                state["last_ost_val"] = int(val)
        except Exception:
            pass
        return val
    except Exception:
        return None

def init_motor():
    send_quiet("EN")
    send_quiet("V0")
    try:
        state["drive_enabled"] = True
        update_drive_status()
    except Exception:
        pass

    send_quiet("V0")

def safe_stop():
    """Soft stop: stop motion without disabling output stage."""
    try:
        send_quiet("V0")
    except Exception:
        pass

def emergency_stop():
    """Hard stop: disable output stage. Requires init_motor() before next move."""
    try:
        send_quiet("DI")
    except Exception:
        pass
    try:
        state["drive_enabled"] = False
        update_drive_status()
        state["force_preload_next"] = True
        update_backlash_status()
    except Exception:
        pass

def recover_after_stop():
    """Re-arm the drive/UI after STOP so Jog/Go work immediately."""
    try:
        state['current_step_target_abs'] = None
        state['stop_flag'] = False
        state['force_preload_next'] = True
        try:
            update_backlash_status()
            update_drive_status()
        except Exception:
            pass
        state['is_paused'] = False
        state['is_scanning'] = False
        if ser_ok():
            s = state['ser']
            try:
                s.reset_input_buffer(); s.reset_output_buffer()
            except Exception:
                pass
            for cmd in ("V0","EN"):
                try: send_cmd(cmd)
                except Exception: pass
        set_fsm("IDLE")
        set_buttons_scanning(False)
        set_buttons_connected(state.get("connected", False))
        log("[OK] Recovered after STOP; ready.")
    except Exception as e:
        log(f"[WARN] Recover failed: {e}", "warn")

def read_position() -> int:
    """Read absolute steps via POS (controller-specific)."""
    try:
        return int(send_cmd("POS", timeout=POLL_TIMEOUT) or "0")
    except Exception as e:
        return 0

def nm_to_steps(nm: float) -> int:
    # Only meaningful as a relative conversion unless you have an absolute offset.
    return int(round(nm * STEPS_PER_NM))

def steps_for_delta_nm(delta_nm: float) -> int:
    steps = int(round(delta_nm * STEPS_PER_NM))
    if INVERT_DIRECTION:
        steps = -steps
    return steps

def read_position_retry(n=3, dt=0.05):
    """Read position with short retries to avoid transient errors."""
    last = None
    for _ in range(max(1, int(n))):
        try:
            return read_position()
        except Exception as e:
            last = e
            time.sleep(dt)
    # last attempt (let it raise if still failing)
    return read_position()


# ---- RS-232 SDO micro-client for Faulhaber (binary frames, CRC8-D5) ----
def _crc8_d5(payload_bytes: bytes) -> int:
    crc = 0xFF
    for b in payload_bytes:
        crc ^= b
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xD5) if (crc & 1) else (crc >> 1)
    return crc & 0xFF

def _frame_sdo_read(node: int, index: int, sub: int) -> bytes:
    idx_lb, idx_hb = index & 0xFF, (index >> 8) & 0xFF
    user = bytes([7, node & 0xFF, 0x01, idx_lb, idx_hb, sub & 0xFF])
    crc = _crc8_d5(user)
    return b"S" + user + bytes([crc]) + b"E"

def _read_sdo_value(index: int, sub: int, node: int, timeout_s=0.3) -> int:
    """Poll one SDO value (unsigned LE int) from 0xXXXX.sub. Raises on timeout."""
    if not ser_ok():
        raise TimeoutError("No serial")
    s = state["ser"]
    frame = _frame_sdo_read(node, index, sub)
    deadline = time.time() + float(timeout_s)
    # Use the same serial lock as ASCII commands to avoid interleaving
    with state["lock"]:
        try:
            s.reset_input_buffer()
            s.reset_output_buffer()
            s.timeout = timeout_s
            s.write(frame); s.flush()
            # Parse frames until we see SDORead response for (index,sub)
            while time.time() < deadline:
                # sync to 'S'
                got = s.read_until(b"S")
                if not got or got[-1:] != b"S":
                    continue
                length = s.read(1)
                if not length:
                    continue
                n = length[0]
                payload = s.read(n)
                eof = s.read(1)
                if len(payload) != n or eof != b"E":
                    continue
                # CRC check
                if _crc8_d5(bytes([n]) + payload[:-1]) != payload[-1]:
                    continue
                node_rx, cmd = payload[0], payload[1]
                if cmd != 0x01:   # not an SDORead response; ignore async frames
                    continue
                idx_lb, idx_hb, sub_rx = payload[2], payload[3], payload[4]
                if (((idx_hb << 8) | idx_lb) != index) or (sub_rx != sub):
                    continue
                raw = payload[5:-1]  # value bytes (1..4 bytes LE)
                return int.from_bytes(raw, "little", signed=False)
        except Exception as e:
            raise
    raise TimeoutError(f"No SDO response for 0x{index:04X}.{sub:02X}")

def _read_statusword() -> int:
    return _read_sdo_value(0x6041, 0x00, STATUSWORD_NODE, timeout_s=0.3)

# ---- Hybrid wait helpers (Statusword primary, POS fallback) ----
def _pos_wait_impl_fallback(target_steps: int, tol_steps: int, stable_polls: int,
                            total_timeout_s: float, t_start: Optional[float] = None,
                            paused_accum_init: float = 0.0) -> bool:
    """Existing POS-tolerance wait semantics, refactored as helper."""
    start = time.time() if t_start is None else t_start
    paused_accum = paused_accum_init
    stable = 0
    while (time.time() - start - paused_accum) < total_timeout_s:
        # Pause handling (freeze timeout while paused)
        if state.get("is_paused"):
            t_pause = time.time()
            while state.get("is_paused") and not state.get("stop_flag"):
                time.sleep(POLL_INTERVAL)
            paused_accum += (time.time() - t_pause)
            continue
        if state.get("stop_flag"):
            return False
        
        # OST guard (errors immediate; input changes debounced; baseline honoured)
        try:
            if (time.time() - last_ost) >= 0.10:
                last_ost = time.time()
                ost = read_ost(timeout=0.2)
                if ost is not None:
                    # 1) error bits -> immediate abort
                    if ost & OST_ERROR_MASK:
                        safe_stop()
                        log(f"[ERROR] OST abort: {_decode_ost_flags(ost)} (OST=0x{ost:04X})", "error")
                        return False
                    # 2) status inputs: only abort if NEW inputs became active vs baseline (debounced 3x)
                    status_bits = ost & OST_INPUT_MASK
                    new_edges = (status_bits ^ (ost_baseline & OST_INPUT_MASK)) & status_bits
                    if new_edges:
                        ost_debounce += 1
                        if ost_debounce >= 3:
                            safe_stop()
                            state['limit_blocked'] = True
                            names = []
                            if new_edges & OST_STATUS_INPUT1: names.append("IN1")
                            if new_edges & OST_STATUS_INPUT2: names.append("IN2")
                            if new_edges & OST_STATUS_INPUT3: names.append("IN3")
                            log(f"[LIMIT] Controller blocked: input change ({'+'.join(names)}) (OST=0x{ost:04X})", "info")
                            return False
                    else:
                        ost_debounce = 0
        except Exception:
            pass

        try:
            pos = read_position_retry()
            if abs(pos - int(target_steps)) <= int(tol_steps):
                stable += 1
                if stable >= int(stable_polls):
                    return True
            else:
                stable = 0
        except Exception:
            stable = 0
        time.sleep(POLL_INTERVAL)
    log(f"[TIMEOUT] Did not reach target {int(target_steps)} in {float(total_timeout_s):.1f}s", "warn")
    return False

def _wait_via_statusword_then_pos(target_steps: int,
                                  tol_steps: int,
                                  stable_polls: int,
                                  total_timeout_s: float) -> bool:
    """
    Robust CiA-402 wait:
      1) ARM: expect bit-12 ACK and bit-10 drop (or tolerate if already in window by POS).
      2) RUN: wait for bit-10 to rise and hold (debounce), optionally cross-check POS.
    Falls back to POS if SDO fails at any point.
    """
    start = time.time()
    paused_accum = 0.0

    # ARM phase — see ACK and TR drop
    saw_ack = False
    saw_tr_drop = False
    t0 = time.time()
    while time.time() - t0 < float(STATUS_ACK_ARM_TIMEOUT_S):
        if state.get("is_paused"):
            t_pause = time.time()
            while state.get("is_paused") and not state.get("stop_flag"):
                time.sleep(STATUS_POLL_S)
            paused_accum += (time.time() - t_pause)
        if state.get("stop_flag"):
            return False

        try:
            sw = _read_statusword()
        except Exception:
            # SDO not available → delegate to POS immediately
            return _pos_wait_impl_fallback(target_steps, tol_steps, stable_polls,
                                           total_timeout_s, start, paused_accum)

        if sw & STATUS_ACK_BIT:
            saw_ack = True
        if not (sw & STATUS_TR_BIT):
            saw_tr_drop = True
        if saw_ack and saw_tr_drop:
            break
        time.sleep(STATUS_POLL_S)

    # If TR never dropped, maybe we were already “in window”
    if not saw_tr_drop:
        try:
            if abs(read_position_retry() - int(target_steps)) <= int(tol_steps):
                return True
        except Exception:
            pass  # continue anyway

    # RUN phase — wait for TR high with debounce; cross-check POS
    tr_hold = 0
    while (time.time() - start - paused_accum) < float(total_timeout_s):
        if state.get("is_paused"):
            t_pause = time.time()
            while state.get("is_paused") and not state.get("stop_flag"):
                time.sleep(STATUS_POLL_S)
            paused_accum += (time.time() - t_pause)
        if state.get("stop_flag"):
            return False

        try:
            sw = _read_statusword()
        except Exception:
            # SDO flaked mid-run → fall back to POS
            return _pos_wait_impl_fallback(target_steps, tol_steps, stable_polls,
                                           total_timeout_s, start, paused_accum)

        if sw & STATUS_FAULT_BIT:
            log("[FAULT] Statusword shows FAULT during motion.", "warn")
            return False

        if sw & STATUS_TR_BIT:
            tr_hold += 1
            if tr_hold >= max(1, int(STATUS_TR_DEBOUNCE_POLLS)):
                # Optional POS sanity cross-check
                try:
                    if abs(read_position_retry() - int(target_steps)) <= int(tol_steps):
                        return True
                except Exception:
                    pass
                return True
        else:
            tr_hold = 0

        time.sleep(STATUS_POLL_S)

    log(f"[TIMEOUT] No TR within {float(total_timeout_s):.1f}s (Statusword path).", "warn")
    return False





# (removed old smart_preload_before_move with old signature)

def wait_until_position(target_steps: int, tol=POS_TOL_STEPS, stable_req=POS_STABLE_COUNT, timeout: Optional[float] = None) -> bool:
    """
    Pause-aware wait for target position.
    - Freezes timeout while paused (so Pause won't cause a timeout).
    - If `timeout` is None, compute it dynamically from remaining distance.
    """
    # Dynamic timeout if not provided
    if timeout is None:
        try:
            pos0 = read_position()
            remaining_nm = abs((int(target_steps) - pos0) / float(STEPS_PER_NM))
            timeout = timeout_for_nm(remaining_nm)
        except Exception:
            timeout = max(MIN_MOVE_TIMEOUT, TIMEOUT_OVERHEAD_S + 1.0)

    # Route: Statusword primary (if enabled), else POS fallback. Always pause-aware.
    if USE_STATUSWORD:
        try:
            return _wait_via_statusword_then_pos(
                target_steps=int(target_steps),
                tol_steps=int(tol),
                stable_polls=int(stable_req),
                total_timeout_s=float(timeout),
            )
        except Exception as e:
            log(f"[WARN] Statusword path failed ({e}); falling back to POS.", "warn")

    return _pos_wait_impl_fallback(
        target_steps=int(target_steps),
        tol_steps=int(tol),
        stable_polls=int(stable_req),
        total_timeout_s=float(timeout),
    )
def _daq_channel_path():
    """
    Build NI-DAQ channel string from current UI state, e.g. "Dev1/ai1".
    Falls back safely if fields are missing.
    """
    try:
        dev = (state.get('daq_dev') or 'Dev1').strip()
    except Exception:
        dev = 'Dev1'
    try:
        ai = (state.get('daq_ai') or 'ai1').strip()
    except Exception:
        ai = 'ai1'
    # sanitize
    try:
        if not re.match(r'^ai\d+$', ai):
            ai = 'ai1'
        if not dev:
            dev = 'Dev1'
    except Exception:
        pass
    return f"{dev}/{ai}"
def read_pmt_voltage(chan=None, samples=10, rate=1000):
    """Return one averaged voltage using NI-DAQ if available; else simulator."""
    if chan is None:
        try:
            chan = _daq_channel_path()
        except Exception:
            chan = "Dev1/ai1"

    if not DAQ_AVAILABLE or nidaqmx is None:
        return 0.2 + 0.05*random.random()

    try:
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(
                chan, min_val=0.0, max_val=10.0,
                terminal_config=TerminalConfiguration.RSE
            )
            if int(samples) > 1:
                task.timing.cfg_samp_clk_timing(int(rate), sample_mode=AcquisitionType.FINITE, samps_per_chan=int(samples))
                vals = task.read(number_of_samples_per_channel=int(samples))
                if isinstance(vals, list) and vals:
                    return float(sum(vals)/len(vals))
                return float(vals) if isinstance(vals, (int,float)) else 0.0
            else:
                return float(task.read())
    except Exception:
        return 0.0
# ------------------ GUI / Logging ------------------
def log(msg, tag="info"):
    # Suppress timeout messages if desired
    try:
        if SUPPRESS_TIMEOUT_MESSAGES and (('[TIMEOUT]' in str(msg)) or ('timeout' in str(msg).lower())):
            return
    except Exception:
        pass
    console.configure(state="normal")
    console.insert(tk.END, msg + "\n", tag)
    console.see(tk.END)
    console.configure(state="disabled")

def set_buttons_connected(connected: bool):
    btn_connect.configure(state=("disabled" if connected else "normal"))
    btn_disconnect.configure(state=("normal" if connected else "disabled"))
    btn_goto.configure(state=("normal" if connected else "disabled"))
    btn_scan.configure(state=("normal" if connected else "disabled"))
    btn_pause.configure(state=("disabled"))
    btn_resume.configure(state=("disabled"))
    btn_stop.configure(state=("normal" if connected else "disabled"))

    # Jog & Queue availability
    try:
        btn_jog_minus.configure(state=("normal" if connected else "disabled"))
        btn_jog_plus.configure(state=("normal" if connected else "disabled"))
        btn_mark_point.configure(state=("normal" if connected else "disabled"))
        btn_add_queue.configure(state=("normal" if connected else "disabled"))
        btn_remove_queue.configure(state=("normal" if connected else "disabled"))
        btn_run_queue.configure(state=("normal" if connected else "disabled"))
        btn_clear_queue.configure(state=("normal" if connected else "disabled"))
    except Exception:
        pass



def set_buttons_scanning(scanning: bool):
    """Disable conflicting controls while scanning and toggle pause/resume."""
    if scanning:
        btn_goto.configure(state="disabled")
        btn_disconnect.configure(state="disabled")
        btn_scan.configure(state="disabled")
        btn_pause.configure(state="normal")
        btn_resume.configure(state="disabled")
        btn_stop.configure(state="normal")
    else:
        set_buttons_connected(state["connected"])

    # Toggle jog/queue while scanning
    try:
        if scanning:
            btn_jog_minus.configure(state="disabled")
            btn_jog_plus.configure(state="disabled")
            btn_mark_point.configure(state="disabled")
            btn_run_queue.configure(state="disabled")
        else:
            btn_jog_minus.configure(state=("normal" if state["connected"] else "disabled"))
            btn_jog_plus.configure(state=("normal" if state["connected"] else "disabled"))
            btn_mark_point.configure(state=("normal" if state["connected"] else "disabled"))
            btn_run_queue.configure(state=("normal" if state["connected"] else "disabled"))
    except Exception:
        pass


# --- Smart Preload: initialization hook ---
preloader = None
def _smart_preload_init():
    global preloader
    try:
        # Try to pick up your existing calibration variables if present; fallback to sane defaults.
        STEPS_PER_NM = globals().get('STEPS_PER_NM', 361765)
        STEPS_PER_NM = globals().get('STEPS_PER_NM', 387650)
        # Slip magnitude per direction in STEPS; adjust in your settings if you have better numbers.
        N_SLIP_FWD_STEPS = int(globals().get('N_SLIP_FWD_STEPS', 68000))
        N_SLIP_BACK_STEPS = int(globals().get('N_SLIP_BACK_STEPS', 199400))

        def read_current_opt_mech_nm():
            # Expect your code to provide these helpers; else adapt below:
            lam_opt = _get_optical_nm()     # GUI/compute
            lam_mech = _get_motor_nm() # from POS and steps_per_nm
            return lam_opt, lam_mech

        def nm_to_steps_signed(nm, dir_):
            # Uses direction-specific steps_per_nm for better accuracy
            spn = STEPS_PER_NM if dir_ == +1 else STEPS_PER_NM
            steps = int(round(nm * spn))
            return steps if dir_ == +1 else -steps

        def send_lr_m(steps):
            serial_write(f"LR{int(steps)};M")

        def wait_until_target_steps(target_steps, timeout_s):
            # Prefer a function already present in your code; else fall back to wait_until_position
            if 'wait_until_target_steps' in globals():
                return globals()['wait_until_target_steps'](target_steps, timeout_s)
            elif 'wait_until_position' in globals():
                # wait_until_position(pos_steps, timeout_s)
                return globals()['wait_until_position'](target_steps, timeout_s)
            else:
                raise RuntimeError("No wait function found (wait_until_target_steps / wait_until_position).")

        def sleep_fn(sec):
            if 'sleep_with_pause' in globals():
                globals()['sleep_with_pause'](sec)
            else:
                import time; time.sleep(sec)

        def get_pos_steps():
            if 'get_current_pos_steps' in globals():
                return globals()['get_current_pos_steps']()
            elif 'read_position_steps' in globals():
                return globals()['read_position_steps']()
            else:
                # As a last resort, query device:
                serial_write("POS?")
                return int(serial_read_int())

        def _log(msg): 
            if 'log' in globals():
                globals()['log'](msg)
            else:
                print(msg)

        def _safe_stop(reason="smart preload stop"):
            if 'safe_stop' in globals():
                globals()['safe_stop'](reason)
            else:
                # Fallback: issue a soft stop
                try:
                    serial_write("ST")
                except Exception:
                    pass
                print("[SAFE_STOP]", reason)

        spd = {+1: float(STEPS_PER_NM), -1: float(STEPS_PER_NM)}
        slip = {+1: int(N_SLIP_FWD_STEPS), -1: int(N_SLIP_BACK_STEPS)}

        # Build preloader
        preloader = SmartPreloader(
            n_slip_steps=slip,
            steps_per_nm_dir=spd,
            read_opt_mech_nm=read_current_opt_mech_nm,
            nm_to_steps=nm_to_steps_signed,
            get_pos_steps=get_pos_steps,
            send_lr_m=send_lr_m,
            wait_until_target=wait_until_target_steps,
            sleep_fn=sleep_fn,
            log_fn=_log,
            safe_stop_fn=_safe_stop,
        )
        # Tune timing (your measured speed)
        preloader.sec_per_nm = 10.0
        preloader.preload_timeout_headroom = 1.6
        preloader.tol_min_nm = 0.03
        preloader.tol_frac_of_slip = 0.10
        _log("[SMART] Preloader initialized.")
    except Exception as e:
        print("[SMART] init failed:", e)
        raise

def on_connect():
    try:
        show_lambda_window()
    except Exception:
        pass
    _smart_preload_init()
    port = port_var.get().strip()
    try:
        baud = int(baud_var.get())
    except Exception as e:
        baud = DEFAULT_BAUD
    try:
        ser = serial.Serial(port, baud, timeout=0.8)
        state["ser"] = ser
        state["connected"] = True
        set_buttons_connected(True)
        set_fsm("IDLE")
        log(f"[OK] Connected {port} @ {baud} baud")
        # Enable replies and verify
        ok = False
        try:
            _ = send_cmd("ANSW1", timeout=1.0)
            probe = send_cmd("VER", timeout=1.5) or send_cmd("POS", timeout=1.5)
            if probe:
                ok = True
        except Exception:
            pass
        if not ok:
            try:
                _ = send_cmd("ANSW2", timeout=1.0)
                probe = send_cmd("VER", timeout=1.5) or send_cmd("POS", timeout=1.5)
                ok = bool(probe)
            except Exception:
                ok = False
        log("[OK] Replies {}abled".format("en" if ok else "NOT en"))
        try:
            send_quiet("ANSW1")
            log("[OK] ANSW1 set (controller replies enabled)")
        except Exception:
            pass
        init_motor()
        log("[OK] Motor initialized (EN, V0)")
    except Exception as e:
        log(f"[ERROR] Could not open {port}: {e}", "error")
        state["ser"] = None
        state["connected"] = False
        set_buttons_connected(False)

        set_fsm("IDLE")
def on_disconnect():
    try:
        if ser_ok():
            safe_stop()
            state["ser"].close()
        state["connected"] = False
        set_buttons_connected(False)
        set_fsm("IDLE")
        log("[OK] Disconnected and motor set safe.")
    except Exception as e:
        log(f"[WARN] Disconnect issue: {e}", "warn")

# ------------------ FSM Actions ------------------
def goto_wavelength_action():
    if state["is_scanning"]:
        messagebox.showwarning("Busy", "Cannot GOTO while scanning.")
        return
    if not ser_ok():
        messagebox.showerror("Error", "Not connected.")
        return
    try:
        tgt = validate_goto_input()
        if tgt is None: return
        target_nm = float(tgt)
    except Exception:
        return
        return

    def goto_worker():
        set_fsm("MOVING")
        try:
            # Lock controls during goto
            btn_goto.configure(state="disabled")
            btn_disconnect.configure(state="disabled")
            btn_scan.configure(state="disabled")
            btn_pause.configure(state="disabled")
            btn_resume.configure(state="disabled")
            btn_stop.configure(state="normal")

            init_motor()
            current_steps = read_position()
            current_nm_ui = float(entry_current.get() or state["current_nm"])
            delta_nm = target_nm - current_nm_ui
            rel_steps = steps_for_delta_nm(delta_nm)
            # target absolute steps estimate:
            target_steps = current_steps + rel_steps
            state['current_step_target_abs'] = target_steps
            maybe_preload_on_reversal(-1 if (delta_nm) < 0 else 1)
            drain_rx_quiet()
            log(f"[MOVE] LR{rel_steps}; M")
            send_quiet(f"LR{rel_steps}")
            send_quiet("M")
            reached = wait_until_position(target_steps, timeout=timeout_for_target_steps(target_steps, delta_nm))
            if SAFE_AFTER_MOVE: routine_stop_pause()
            if reached:
                try:
                    nm_dir = -1 if (target_nm - current_nm_ui) < 0 else (1 if (target_nm - current_nm_ui) > 0 else 0)
                    if nm_dir != 0:
                        state['last_nm_dir'] = int(nm_dir)
                        state['force_preload_next'] = False
                        update_backlash_status(nm_dir)
                except Exception:
                    pass
                state["current_nm"] = target_nm
                entry_current.delete(0, tk.END); entry_current.insert(0, f"{state['current_nm']:.3f}")
                if not postmove_limit_audit('goto'):
                    log("[STOP] Drive reported fault/limit after move; halting.", "warn")
                else:
                    log(f"[DONE] Reached {target_nm:.3f} nm")


            else:
                log("[STOP] Motion interrupted")
                # Auto-recover once and retry the goto
                recover_after_stop()
                init_motor()
                current_steps = read_position()
                try:
                    current_nm_ui = float(entry_current.get() or state["current_nm"])
                except Exception:
                    current_nm_ui = state.get("current_nm", 0.0)
                delta_nm = target_nm - current_nm_ui
                rel_steps = steps_for_delta_nm(delta_nm)
                target_steps = current_steps + rel_steps
                state['current_step_target_abs'] = target_steps
                drain_rx_quiet()
                log(f"[MOVE] LR{rel_steps}; M")
                send_quiet(f"LR{rel_steps}")
                send_quiet("M")
                reached2 = wait_until_position(target_steps, timeout=timeout_for_target_steps(target_steps, abs(delta_nm)))
                if SAFE_AFTER_MOVE: routine_stop_pause()
                if reached2:
                    state["current_nm"] = target_nm
                    try:
                        entry_current.delete(0, tk.END); entry_current.insert(0, f"{state['current_nm']:.3f}")
                    except Exception:
                        pass
                    log(f"[DONE] Reached {target_nm:.3f} nm (after auto-recover)")
                else:
                    log("[ERROR] Goto failed after auto-recover.", "error")
        except Exception as e:
            log(f"[ERROR] goto_wavelength: {e}", "error")
            set_fsm("ERROR")
        finally:
            # Restore controls and FSM back to IDLE when done
            set_buttons_connected(state["connected"])
            if state.get("fsm") == "MOVING":
                set_fsm("IDLE")

    threading.Thread(target=goto_worker, daemon=True).start()

def _get_presettle_sec():
    try:
        return max(0.0, float(presettle_ms_var.get())/1000.0)
    except Exception:
        return 0.0

def _get_avg_fraction():
    try:
        return max(0.0, min(100.0, float(avg_fraction_var.get())))
    except Exception:
        return 100.0


def sleep_with_pause(seconds: float):
    """Sleep up to 'seconds' while honoring pause/stop."""
    end_t = time.time() + max(0.0, float(seconds))
    while time.time() < end_t:
        if state.get("stop_flag"):
            break
        if state.get("is_paused"):
            time.sleep(0.05)
            continue
        time.sleep(min(0.05, max(0.0, end_t - time.time())))

def read_pmt_voltage_avg(samples: int = None, window_sec: float = None):
    """Average DAQ reading over N samples; N from UI if not provided.
    This is a tight loop of read_pmt_voltage(); it respects stop/pause.
    Returns the arithmetic mean (or a single read if interrupted)."""
    # If window_sec is provided, average over that duration (seconds)
    if window_sec is not None and window_sec > 0:
        t0 = time.time()
        total = 0.0
        count = 0
        while (time.time() - t0) < float(window_sec) and not state.get("stop_flag"):
            while state.get("is_paused") and not state.get("stop_flag"):
                time.sleep(0.05)
            v = read_pmt_voltage()
            total += float(v)
            count += 1
        if count <= 0:
            return float(read_pmt_voltage())
        return total / count

    # Otherwise: average a fixed number of samples
    try:
        if samples is None:
            samples = int(avg_samples_var.get())
    except Exception:
        samples = 1
    samples = max(1, int(samples))

    total = 0.0
    count = 0
    for _ in range(samples):
        if state.get("stop_flag"):
            break
        while state.get("is_paused") and not state.get("stop_flag"):
            time.sleep(0.05)
        v = read_pmt_voltage()
        total += float(v)
        count += 1
    if count <= 0:
        return float(read_pmt_voltage())
    return total / count
def scan_action():
    
    if state["is_scanning"]:
        messagebox.showwarning("Busy", "Scan already running.")
        return
    if not ser_ok():
        if bool(globals().get("SIM_ALLOW_SCAN_WHEN_DISCONNECTED", True)):
            log("[SIM] Not connected; running scan in simulation mode.", "warn")
        else:
            messagebox.showerror("Error", "Not connected.")
            return
    _cfg = validate_scan_inputs()
    if _cfg is None:
        return
    s_nm = _cfg["start"]; e_nm = _cfg["end"]; st_nm = _cfg["step"]; dwell = _cfg["dwell"]
    state['planned_params'] = {'s_nm': s_nm, 'e_nm': e_nm, 'st_nm': st_nm, 'dwell': dwell}
    log(f"[SCAN] start={s_nm:.3f} end={e_nm:.3f} step={st_nm:.4f} dwell={dwell:.3f}s")



    direction = 1 if e_nm >= s_nm else -1
    state['planned_params']['direction'] = (1 if e_nm >= s_nm else -1)
    state["is_scanning"] = True
    set_fsm("SCAN")
    set_buttons_scanning(True)
    state["is_paused"] = False
    state["stop_flag"] = False

    btn_pause.configure(state="normal")
    btn_stop.configure(state="normal")
    btn_resume.configure(state="disabled")

    def scan_worker():
        try:
            init_motor()
            state['force_preload_next'] = True
            # Move to start
            current_steps = read_position()
            current_nm_ui = float(entry_current.get() or state["current_nm"])
            # First jump: from current UI nm to start nm
            delta_first = s_nm - current_nm_ui
            rel_first = steps_for_delta_nm(delta_first)
            target_first = current_steps + rel_first
            state['current_step_target_abs'] = target_first
            maybe_preload_on_reversal(-1 if (s_nm - current_nm_ui) < 0 else 1)
            log(f"[MOVE] To start {s_nm:.3f} nm (Δ={delta_first:+.3f} nm, steps={rel_first})")
            send_quiet(f"LR{rel_first}"); send_quiet("M")
            if not wait_until_position(target_first, timeout=timeout_for_target_steps(target_first, delta_first)):
                safe_stop(); raise RuntimeError("Stopped before reaching start.")
            state["current_nm"] = s_nm
            entry_current.delete(0, tk.END); entry_current.insert(0, f"{state['current_nm']:.3f}")
            if SAFE_AFTER_MOVE: routine_stop_pause()

            # Prepare plotting via PlotManager
            label = time.strftime('%Y-%m-%d %H:%M') + f" λ={s_nm:.3f}→{e_nm:.3f} Δ={st_nm:.3f}nm"
            state['active_plot_id'] = plot_mgr.start_new_scan(label, {'start': s_nm, 'end': e_nm, 'step': st_nm})

            pos_nm = s_nm
            rel_step_steps = steps_for_delta_nm(st_nm) * direction
            while (direction > 0 and pos_nm <= e_nm + 1e-9) or (direction < 0 and pos_nm >= e_nm - 1e-9):
                if state["stop_flag"]:
                    safe_stop(); log("[STOP] Scan aborted"); break

                # Measure
                v = read_pmt_voltage()
                if state.get('active_plot_id'):
                    plot_mgr.append_scan_point(state['active_plot_id'], pos_nm, v)
                log(f"{pos_nm:.3f} nm  |  {v:.5f} V")

                # Next step
                pos_nm_next = pos_nm + st_nm*direction
                                # Predict target steps from current absolute (before issuing move)
                curr_abs = read_position()
                target_abs = curr_abs + rel_step_steps
                state['current_step_target_abs'] = target_abs
                maybe_preload_on_reversal(direction)
                drain_rx_quiet()
                send_quiet(f"LR{rel_step_steps}")
                send_quiet("M")

                # Pause handling during move
                while state["is_paused"] and not state["stop_flag"]:
                    time.sleep(0.05)

                if not wait_until_position(target_abs, timeout=timeout_for_target_steps(target_abs, st_nm)):
                    safe_stop(); log("[STOP] Scan interrupted"); break
                # Post-step limit audit (passive; stops only on FAULT/INTERNAL_LIMIT)
                try:
                    if not postmove_limit_audit('scan-step'):
                        safe_stop()
                        log("[STOP] Scan halted due to drive limit/fault (post-step audit).", "warn")
                        break
                except Exception:
                    pass


                if SAFE_AFTER_MOVE: routine_stop_pause()

                pos_nm = pos_nm_next
                state["current_nm"] = pos_nm
                state['planned_params']['last_nm'] = pos_nm
                entry_current.delete(0, tk.END); entry_current.insert(0, f"{state['current_nm']:.3f}")

                # Pause checkpoint: after completing this step
                if state.get('pause_requested'):
                    state['pause_requested'] = False
                    state['is_paused'] = True
                    set_fsm('PAUSED')
                    try:
                        btn_pause.configure(state='disabled'); btn_resume.configure(state='normal')
                    except Exception:
                        pass
                    if ser_ok():
                        safe_stop()
                    log('[PAUSE] Paused after finishing current step.')
                    return
                time.sleep(dwell)

            state["is_scanning"] = False
            btn_pause.configure(state="disabled"); btn_resume.configure(state="disabled")
            log("[DONE] Scan complete.")
            try:
                if state.get('active_plot_id'):
                    plot_mgr.finalize_scan(state['active_plot_id'], status="completed")

                try:
                    if autosave_csv_var.get():
                        export_active_scan_csv()
                except Exception:
                    pass

            except Exception:
                pass
            set_fsm("IDLE")
        except Exception as e:
            state["is_scanning"] = False
            btn_pause.configure(state="disabled"); btn_resume.configure(state="disabled")
            log(f"[ERROR] scan: {e}", "error")
            try:
                if state.get('active_plot_id'):
                    plot_mgr.finalize_scan(state['active_plot_id'], status="error")
            except Exception:
                pass

    threading.Thread(target=scan_worker, daemon=True).start()

def pause_action():
    """Pause after current motion step completes (no immediate stop)."""
    if not state["is_scanning"]:
        return
    try:
        state['pause_requested'] = True
        try:
            btn_pause.configure(state="disabled"); btn_resume.configure(state="disabled")
        except Exception:
            pass
        log("[PAUSE] Will pause after the current step finishes.")
    except Exception as e:
        log(f"[ERROR] pause: {e}", "error")

def resume_action():
    """Resume scan without re-reading GUI. Uses stored planned_params to finish remaining points."""
    if not state.get('is_scanning'):
        log('[RESUME] Ignored: no active scan.', 'warn')
        return
    if not state.get('is_paused'):
        log('[RESUME] Ignored: not paused.', 'warn')
        return
    threading.Thread(target=do_resume, daemon=True).start()

def do_resume():
    """Resume a paused scan using stored planned_params only (no GUI re-read)."""
    try:
        if not state.get("is_scanning"):
            return
        state['pause_requested'] = False
        if not ser_ok():
            messagebox.showerror("Error", "Not connected.")
            return
        state["is_paused"] = False
        set_fsm("SCAN")
        set_buttons_scanning(True)
        try:
            btn_pause.configure(state="normal"); btn_resume.configure(state="disabled"); btn_stop.configure(state="normal")
        except Exception:
            pass

        p = state.get('planned_params', {}) or {}
        s_nm = float(p.get('s_nm')); e_nm = float(p.get('e_nm')); st_nm = float(p.get('st_nm')); dwell = float(p.get('dwell')); direction = int(p.get('direction', 1))

        init_motor()

        pos_nm = float((p.get('last_nm') if p.get('last_nm') is not None else state.get('current_nm', s_nm)))
        tgt_abs = state.get("current_step_target_abs", None)
        if ser_ok() and tgt_abs is not None:
            # PATCH: Do NOT finish pending target on resume; clear it to avoid pre-move.
            state["current_step_target_abs"] = None
            log("[RESUME] Ignoring pending target; continuing from current position.", "warn")
            try:
                # Keep UI current wavelength as-is (no pre-move)
                entry_current.delete(0, tk.END); entry_current.insert(0, f"{pos_nm:.3f}")
            except Exception:
                pass
        if direction > 0 and pos_nm < s_nm - 1e-12: pos_nm = s_nm
        if direction < 0 and pos_nm > s_nm + 1e-12: pos_nm = s_nm

        rel_step_steps = steps_for_delta_nm(st_nm) * direction

        while (direction > 0 and pos_nm <= e_nm + 1e-12) or (direction < 0 and pos_nm >= e_nm - 1e-12):
            if state.get("stop_flag"):
                safe_stop(); log("[STOP] Resume aborted", "warn"); break

            presettle = _get_presettle_sec()
            if presettle > 0:
                sleep_with_pause(min(presettle, dwell))
            t_meas = time.time()
            if avg_mode_var.get() == 'time':
                frac = _get_avg_fraction()/100.0
                window = max(0.0, min(dwell - min(presettle, dwell), dwell * frac))
                v = read_pmt_voltage_avg(window_sec=window)
            else:
                v = read_pmt_voltage_avg()
            meas_dur = time.time() - t_meas
            sleep_with_pause(max(0.0, dwell - min(presettle, dwell) - meas_dur))

            try:
                if state.get('active_plot_id'):
                    plot_mgr.append_scan_point(state['active_plot_id'], pos_nm, v)
            except Exception:
                pass
            log(f"{pos_nm:.3f} nm  |  {v:.5f} V")

            pos_nm_next = pos_nm + st_nm*direction
            curr_abs = read_position()
            target_abs = curr_abs + rel_step_steps
            state["current_step_target_abs"] = target_abs
            maybe_preload_on_reversal(direction)
            send_quiet(f"LR{rel_step_steps}"); send_quiet("M")

            while state.get("is_paused") and not state.get("stop_flag"):
                time.sleep(0.05)

            reached = wait_until_position(target_abs, timeout=timeout_for_target_steps(target_abs, abs(st_nm)))
            if not reached:
                safe_stop(); log("[STOP] Resume interrupted", "warn"); break

            if SAFE_AFTER_MOVE: routine_stop_pause()

            try:
                state['last_nm_dir'] = int(direction)
                update_backlash_status(direction)
            except Exception:
                pass

            pos_nm = pos_nm_next
            state['planned_params']['last_nm'] = pos_nm
            state["current_nm"] = pos_nm
            try:
                entry_current.delete(0, tk.END); entry_current.insert(0, f"{state['current_nm']:.3f}")
            except Exception:
                pass

            # Pause checkpoint in resume loop: after finishing this step
            if state.get('pause_requested'):
                state['pause_requested'] = False
                state['is_paused'] = True
                set_fsm('PAUSED')
                try:
                    btn_pause.configure(state='disabled'); btn_resume.configure(state='normal')
                except Exception:
                    pass
                if ser_ok():
                    safe_stop()
                log('[PAUSE] Paused after finishing current step.')
                return
        state["is_scanning"] = False
        set_buttons_scanning(False)
        try:
            if state.get('active_plot_id'):
                plot_mgr.finalize_scan(state['active_plot_id'], status=("interrupted" if state.get("stop_flag") else "completed"))
        except Exception:
            pass
        set_fsm("IDLE")
    except Exception as e:
        state["is_scanning"] = False
        set_buttons_scanning(False)
        log(f"[RESUME ERROR] {e}", "error")
def stop_action():
    state["stop_flag"] = True
    routine_stop_pause()
    state["is_paused"] = False
    state["is_scanning"] = False
    btn_pause.configure(state="disabled"); btn_resume.configure(state="disabled")
    log("[STOP] Stop requested; motor set safe.")
    try:
        if state.get('is_scanning') and state.get('active_plot_id'):
            plot_mgr.finalize_scan(state['active_plot_id'], status="stopped")
    except Exception:
        pass
    set_buttons_connected(state["connected"])
    set_fsm("IDLE")

    recover_after_stop()

def jog_minus_action():
    try:
        step = float(jog_step_var.get())
    except Exception:
        step = 0.5
    try:
        cur_nm = float(entry_current.get() or state.get("current_nm", 0.0))
    except Exception:
        cur_nm = state.get("current_nm", 0.0)
    target = cur_nm - step
    try:
        entry_goto.delete(0, "end")
        entry_goto.insert(0, f"{target:.3f}")
    except Exception:
        pass
    goto_wavelength_action()

def jog_plus_action():
    try:
        step = float(jog_step_var.get())
    except Exception:
        step = 0.5
    try:
        cur_nm = float(entry_current.get() or state.get("current_nm", 0.0))
    except Exception:
        cur_nm = state.get("current_nm", 0.0)
    target = cur_nm + step
    try:
        entry_goto.delete(0, "end")
        entry_goto.insert(0, f"{target:.3f}")
    except Exception:
        pass
    goto_wavelength_action()

def mark_point_action():
    try:
        x_nm = float(entry_current.get() or state.get("current_nm", 0.0))
    except Exception:
        x_nm = state.get("current_nm", 0.0)
    try:
        y_val = (read_pmt_voltage_avg(window_sec=float(entry_wait.get())) if avg_mode_var.get()=='time' else read_pmt_voltage_avg())
    except Exception:
        y_val = None
    if y_val is None:
        log("[WARN] No DAQ reading available for Mark Point.", "warn")
        return
    try:
        plot_mgr.append_jog_point(x_nm, y_val)
        plot_mgr._schedule_redraw()
        log(f"[JOG] Marked point λ={x_nm:.3f} nm, V={y_val:.5f}")
    except Exception as e:
        log(f"[WARN] Could not append jog point: {e}", "warn")

def add_scan_to_queue():
    try:
        s_nm = float(entry_start.get())
        e_nm = float(entry_end.get())
        st_nm = float(entry_step.get())
    except Exception:
        messagebox.showerror("Invalid", "Please enter valid Start/End/Step.")
        return
    item = {"start": s_nm, "end": e_nm, "step": st_nm}
    state["scan_queue"].append(item)
    try:
        queue_listbox.insert("end", f"{len(state['scan_queue']):02d}: {s_nm:.3f} → {e_nm:.3f}, Δ {st_nm:.3f} nm")
    except Exception:
        pass
    log(f"[QUEUE] Added scan: {item}")

def remove_selected_scan():
    try:
        sel = queue_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        queue_listbox.delete(idx)
        try:
            del state["scan_queue"][idx]
        except Exception:
            pass
        for i in range(queue_listbox.size()):
            txt = queue_listbox.get(i)
            rest = txt.split(":", 1)[-1].strip()
            queue_listbox.delete(i)
            queue_listbox.insert(i, f"{i+1:02d}: {rest}")
        log("[QUEUE] Removed selected scan.")
    except Exception as e:
        log(f"[WARN] Remove failed: {e}", "warn")

def clear_scan_queue():
    state["scan_queue"].clear()
    try:
        queue_listbox.delete(0, "end")
    except Exception:
        pass
    log("[QUEUE] Cleared.")

def run_scan_queue_action():
    if state.get("is_scanning"):
        messagebox.showwarning("Busy", "Already scanning.")
        return
    if not state["scan_queue"]:
        messagebox.showinfo("Queue", "Queue is empty.")
        return
    threading.Thread(target=run_scan_queue_worker, daemon=True).start()


# Prompt to save all scans + summary at queue end
try:
    if _queue_finished_prompt("Queue finished", "Save all scans and queue summary?"):
        rows = []
        try:
            pid = state.get('active_plot_id')
            p = state.get('planned_params', {}) or {}
            scanid = state.get("scan_id") or (uuid.uuid4().hex[:8] if 'uuid' in globals() else "")
            saved_path = export_scan_csv_by_id(pid, p, export_dir_var.get(), pattern_var.get(), scanid, state.get("queue_id") if 'state' in globals() else None)
            rows.append(dict(queue_id=state.get("queue_id") if 'state' in globals() else "",
                             scan_id=scanid, filepath=saved_path,
                             start_nm=p.get("s_nm"), end_nm=p.get("e_nm"),
                             step_nm=p.get("st_nm"), wait_s=p.get("dwell"),
                             n_points=(len(plot_mgr.get_points(pid)[0]) if pid else 0),
                             started_at=(state.get("scan_started_at") if 'state' in globals() else ""),
                             finished_at=datetime.datetime.now().isoformat(timespec="seconds"),
                             status=("interrupted" if (state.get("stop_flag") if 'state' in globals() else False) else "completed")))
        except Exception:
            pass
        if rows:
            dirp = export_dir_var.get()
            os.makedirs(dirp, exist_ok=True)
            qid = state.get("queue_id") if 'state' in globals() else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            summ = os.path.join(dirp, f"queue_{qid}_summary.csv")
            with open(summ, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["queue_id","scan_id","filepath","start_nm","end_nm","step_nm","wait_s","n_points","started_at","finished_at","status"])
                for r in rows:
                    w.writerow([r.get("queue_id"), r.get("scan_id"), r.get("filepath"), r.get("start_nm"), r.get("end_nm"),
                                r.get("step_nm"), r.get("wait_s"), r.get("n_points"), r.get("started_at"), r.get("finished_at"), r.get("status")])
            try:
                log(f"[CSV] Queue summary saved: {summ}")
            except Exception:
                pass
except Exception:
    pass

def run_scan_queue_worker():
    try:
        btn_run_queue.configure(state="disabled")
        btn_add_queue.configure(state="disabled")
        btn_remove_queue.configure(state="disabled")
        btn_clear_queue.configure(state="disabled")
    except Exception:
        pass
    for i, item in enumerate(list(state["scan_queue"])):
        if not ser_ok():
            log("[QUEUE] Stopped: not connected.", "warn")
            break
        try:
            entry_start.delete(0, "end"); entry_start.insert(0, f"{item['start']:.3f}")
            entry_end.delete(0, "end"); entry_end.insert(0, f"{item['end']:.3f}")
            entry_step.delete(0, "end"); entry_step.insert(0, f"{item['step']:.3f}")
        except Exception:
            pass
        log(f"[QUEUE] Running {i+1}/{len(state['scan_queue'])}: {item}")
        scan_action()
        while state.get("is_scanning"):
            if state.get("stop_flag"):
                log("[QUEUE] Aborted by Stop.", "warn")
                break
            time.sleep(0.2)
        if state.get("stop_flag"):
            break
    try:
        btn_run_queue.configure(state="normal")
        btn_add_queue.configure(state="normal")
        btn_remove_queue.configure(state="normal")
        btn_clear_queue.configure(state="normal")
    except Exception:
        pass
    log("[QUEUE] Done.")
    try:
        root.after(0, _maybe_prompt_save_all_scans_after_queue)
    except Exception:
        pass



def clear_plot_action():
    try:
        plot_mgr.clear_all()
    except Exception as e:
        log(f"[ERROR] clear plot: {e}", "error")
# ------------------ GUI Build ------------------

def _fmt_nm_dir(n):
    return 'forward' if int(n) > 0 else ('backward' if int(n) < 0 else 'idle')

def update_backlash_status(nm_dir_hint=None):
    try:
        if nm_dir_hint is None:
            nm_dir_hint = int(state.get('last_nm_dir', 0) or 0)
        loaded = (nm_dir_hint < 0) and (not state.get('force_preload_next', False)) and int(state.get('last_nm_dir', 0) or 0) == -1
        dir_txt = _fmt_nm_dir(nm_dir_hint)
        ts = state.get('last_preload_ts')
        age = f", t-{(time.time()-ts):.1f}s" if ts else ""
        if nm_dir_hint < 0:
            txt = f"Backlash ({dir_txt}): {'LOADED' if loaded else 'NEEDS PRELOAD'}{age}"
        elif nm_dir_hint > 0:
            txt = f"Backlash ({dir_txt}): n/a"
        else:
            txt = "Backlash: idle"
        try:
            lbl_backlash.configure(text=txt, fg=("#228B22" if loaded else "#B22222") if nm_dir_hint < 0 else "#111")
        except Exception:
            pass
    except Exception:
        pass

def update_drive_status():
    try:
        en = bool(state.get('drive_enabled', False))
        txt = f"Drive: {'ENABLED' if en else 'DISABLED'}"
        try:
            lbl_drive_status.configure(text=txt, fg=("#228B22" if en else "#B22222"))
        except Exception:
            pass
    except Exception:
        pass

try:
    update_backlash_status()
    update_drive_status()
except Exception:
    pass

root = tk.Tk()
root.title("Monochromator Controller")
root.geometry("1280x800")





# --- Stretchy grid layout ---
try:
    main.grid(row=0, column=0, sticky="nsew")
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    # Two columns: left controls fixed min width; right expands
    main.grid_rowconfigure(0, weight=1)
    main.grid_columnconfigure(0, weight=0)
    main.grid_columnconfigure(0, minsize=340)
    main.grid_columnconfigure(1, weight=1)
    main.grid_columnconfigure(1, weight=1)
except Exception as e:
    try:
        log(f"[UI] Grid weights setup warning: {e}", "warn")
    except Exception:
        pass


# --- Menubar (Settings + View) ---
try:
    menubar = (root["menu"] if root["menu"] else tk.Menu(root))
except Exception:
    menubar = tk.Menu(root)
# Define toggle here (safe to reference 'console' at call time)
def _toggle_log():
    try:
        w = console
    except NameError:
        return
    try:
        info = w.grid_info()
        if not info:
            w.grid(row=2, column=0, sticky="ew")
        else:
            w.grid_remove()
    except Exception:
        pass
try:
    m_settings = tk.Menu(menubar, tearoff=0)
    m_settings.add_command(label="Calibration…", command=lambda: open_calibration_window())
    menubar.add_cascade(label="Settings", menu=m_settings)

    m_view = tk.Menu(menubar, tearoff=0)
    m_view.add_command(label="Toggle Log", command=_toggle_log)
    menubar.add_cascade(label="View", menu=m_view)

    root.config(menu=menubar)
    root.menuname = str(menubar)
except Exception as e:
    try: log(f"[UI] Menubar not created: {e}", "warn")
    except Exception: pass

main = ttk.Frame(root, padding=8)
main.grid(row=0, column=0, sticky="nsew")

main.columnconfigure(0, weight=1)
main.columnconfigure(1, weight=1)


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

left_scroll = ScrollableFrame(main)
left_scroll.grid(row=0, column=0, sticky="nsew")
left = left_scroll.inner
right = ttk.Frame(main)

right.grid(row=0, column=1, sticky="nsew")

# --- Correct grid weights (applied after widgets exist) ---
try:
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    main.grid_rowconfigure(0, weight=1)
    main.grid_columnconfigure(0, weight=0)
    try:
        main.grid_columnconfigure(0, minsize=340)
    except Exception:
        pass
    main.grid_columnconfigure(1, weight=1)
    right.grid_columnconfigure(0, weight=1)
    right.grid_rowconfigure(0, weight=1)
except Exception as _e:
    pass


# --- Right panel weights (plot grows, log visible) ---
try:
    right.grid_rowconfigure(0, weight=1)  # plot
    right.grid_rowconfigure(1, weight=0)  # log
    
    right.grid_rowconfigure(2, weight=0)
    right.grid_columnconfigure(0, weight=1)
except Exception:
    pass

# (log_text removed; using unified 'console' ScrolledText)



# ---- Software # [soft-limit UI removed]


# ---- DAQ Channel selection (ai0–ai10) ----
try:
    ttk.Separator(left).grid(row=38, column=0, columnspan=2, sticky="ew", pady=(8,4))
    ttk.Label(left, text="DAQ device").grid(row=39, column=0, sticky="w")
    if 'daq_dev_var' not in globals():
        daq_dev_var = tk.StringVar(value=state.get('daq_dev', 'Dev1'))
        globals()['daq_dev_var'] = daq_dev_var
    else:
        daq_dev_var = globals()['daq_dev_var']
    entry_daq_dev = ttk.Entry(left, textvariable=daq_dev_var, width=12)
    entry_daq_dev.grid(row=39, column=1, sticky="ew")

    def _on_dev_change(*_):
        try:
            state['daq_dev'] = daq_dev_var.get()
        except Exception:
            pass
    try:
        daq_dev_var.trace_add('write', lambda *_: _on_dev_change())
    except Exception:
        # Tk on older versions: trace method name is 'trace'
        daq_dev_var.trace('w', lambda *_: _on_dev_change())

    ttk.Label(left, text="Analog input").grid(row=40, column=0, sticky="w")
    channels = [f"ai{i}" for i in range(0, 11)]
    if 'daq_ai_var' not in globals():
        daq_ai_var = tk.StringVar(value=state.get('daq_ai', 'ai1'))
        globals()['daq_ai_var'] = daq_ai_var
    else:
        daq_ai_var = globals()['daq_ai_var']
    combo_ai = ttk.Combobox(left, values=channels, textvariable=daq_ai_var, state="readonly", width=12)
    combo_ai.grid(row=40, column=1, sticky="ew")

    def _on_ai_change(*_):
        try:
            state['daq_ai'] = daq_ai_var.get()
        except Exception:
            pass
    try:
        daq_ai_var.trace_add('write', lambda *_: _on_ai_change())
    except Exception:
        daq_ai_var.trace('w', lambda *_: _on_ai_change())
except Exception as e:
    try:
        log(f"[UI] DAQ channel UI init failed: {e}", "warn")
    except Exception:
        print(f"[UI] DAQ channel UI init failed: {e}")

# ---- CSV Export controls ----
try:
    ttk.Label(left, text="Auto-save CSV after scan").grid(row=41, column=0, sticky="w")
    autosave_csv_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(left, variable=autosave_csv_var).grid(row=41, column=1, sticky="w")

    ttk.Label(left, text="Folder").grid(row=42, column=0, sticky="w")
    export_dir_var = tk.StringVar(value=os.path.expanduser("~/scans"))
    entry_export_dir = ttk.Entry(left, textvariable=export_dir_var, width=20)
    entry_export_dir.grid(row=42, column=1, sticky="ew")
    def browse_export_dir():
        d = fd.askdirectory() or ""
        if d: export_dir_var.set(d)
    ttk.Button(left, text="Browse…", command=browse_export_dir).grid(row=43, column=1, sticky="e", padx=(0,0))

    ttk.Label(left, text="Filename pattern").grid(row=43, column=0, sticky="w")
    pattern_var = tk.StringVar(value="scan_{date}_{time}_{start}-{end}nm_{mode}.csv")
    entry_pattern = ttk.Entry(left, textvariable=pattern_var, width=20)
    entry_pattern.grid(row=43, column=1, sticky="ew")

    ttk.Button(left, text="Save CSV now", command=lambda: export_active_scan_csv()).grid(row=44, column=0, columnspan=2, sticky="ew", pady=(6,0))
except Exception:
    pass

# ---- Inserted Slip Compensation

# --- Fallback button to open Calibration (in case menubar is hidden) ---
try:
    ttk.Button(left, text="Calibration…", command=open_calibration_window).grid(sticky="ew", pady=(6,0))
except Exception:
    pass

# ---- UI (relocated after 'left' is created) ----
# ---- Slip Compensation UI ----
try:
    slip_frame = ttk.LabelFrame(left, text="Slip Compensation")
    slip_frame.grid(row=45, column=0, columnspan=2, sticky="ew", pady=(6,2))
    ttk.Label(slip_frame, text="Slip preload (nm):").grid(row=0, column=0, sticky="w")
    slip_nm_var = tk.StringVar(value=str(state.get('n_slip_nm_back', 1.1618)))
    entry_slip_nm = ttk.Entry(slip_frame, textvariable=slip_nm_var, width=12)
    entry_slip_nm.grid(row=0, column=1, sticky="ew", padx=(6,0))

    # Show computed steps for back direction
    _lbl_steps = ttk.Label(slip_frame, text="≈ — steps")
    _lbl_steps.grid(row=0, column=2, sticky="w", padx=(8,0))
    def _refresh_steps_label():
        try:
            spn_back = float(globals().get('STEPS_PER_NM', 361765))
            nm = float(slip_nm_var.get())
            _lbl_steps.config(text=f"≈ {int(round(nm*spn_back))} steps")
        except Exception:
            _lbl_steps.config(text="≈ — steps")
    slip_nm_var.trace_add('write', lambda *a: _refresh_steps_label())
    _refresh_steps_label()
    
    def _apply_n_slip_from_ui():
        try:
            nm_val = float(slip_nm_var.get())
            if nm_val < 0: nm_val = 0.0
            spn_back = float(globals().get('STEPS_PER_NM', 361765))
            steps_val = int(round(nm_val * spn_back))
            state['n_slip_nm_back'] = nm_val
            state['n_slip_steps_back'] = steps_val
            # Backward compatibility save (steps)
            try:
                _save_n_slip(steps_val)
            except Exception:
                pass
            log(f"[SLIP] Back preload set to {nm_val:.4f} nm (~{steps_val} steps back)")
            _refresh_steps_label()
        except Exception as e:
            log(f"[SLIP] Invalid slip nm entry: {e}", "warn")

        btn_save_nslip = ttk.Button(slip_frame, text="Save", command=_apply_n_slip_from_ui)
        btn_save_nslip.grid(row=0, column=2, padx=(6,0))
        # Backlash and Drive indicators
        lbl_backlash = tk.Label(slip_frame, text="Backlash: —", anchor="w")
        lbl_backlash.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4,0))
        lbl_drive_status = tk.Label(slip_frame, text="Drive: —", anchor="w")
        lbl_drive_status.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(2,0))
except Exception:
    pass
except Exception as e:
    try:
        log(f"[SLIP] Could not build Slip UI: {e}", "warn")
    except Exception:
        pass





except Exception:
    pass

# --- Optical/Motor λ helpers & "Set to Current Wavelength" action ---
def _get_optical_nm():
    try:
        v = float(entry_current.get())
        return v
    except Exception:
        return float(state.get('current_nm', 0.0))

def _get_motor_nm():
    try:
        anchor_nm = float(state.get('lambda_anchor_nm', _get_optical_nm()))
        anchor_pos = int(state.get('anchor_pos_steps', read_position()))
        cur_pos = int(read_position())
        steps_per_nm = float(globals().get('STEPS_PER_NM', 361765))
        inv = bool(globals().get('INVERT_DIRECTION', True))
        delta_steps = cur_pos - anchor_pos
        delta_nm = delta_steps / steps_per_nm
        if inv:
            delta_nm = -delta_nm
        return anchor_nm + delta_nm
    except Exception:
        return float(state.get('current_nm', 0.0))

def set_to_current_wavelength_action():
    try:
        src = (entry_current.get().strip() if 'entry_current' in globals() else '') or \
              (entry_goto.get().strip() if 'entry_goto' in globals() else '')
        lam = float(src) if src else float(state.get('current_nm', 0.0))
    except Exception:
        try:
            messagebox.showerror("Input", "Enter a valid wavelength (nm) in 'Current λ' or 'Go To'.")
        except Exception:
            pass
        return
    # 1) Commit the optical wavelength value
    state['current_nm'] = lam
    try:
        entry_current.delete(0, tk.END); entry_current.insert(0, f"{lam:.3f}")
    except Exception:
        pass

    # 2) Anchor the motor reference at the current hardware POS and mark it as POS0 (soft)
    try:
        pos = int(read_position())
        state['lambda_anchor_nm'] = lam            # "optical at anchor"
        state['anchor_pos_steps'] = pos            # "motor at anchor"
        state['pos0_steps'] = pos                  # remember this as our soft POS0
        # NOTE: We do NOT hard-reset the device counter since no 'POS=0' command is defined here.
        # The display will treat this anchor as POS0 so motor λ equals optical λ at press time.
        log(f"[ANCHOR] Set to current wavelength @ POS0: {lam:.3f} nm (soft POS0 at device POS {pos})")
    except Exception as e:
        log(f"[ANCHOR] Failed to anchor: {e}", "warn")

    # Update live labels (if present)
    try:
        if "gui" in globals():
            gui.update_optical_lambda(lam)
            gui.update_motor_lambda(lam)
    except Exception:
        pass


    # 3) Force-refresh any displays and open the 2-field Lambda Monitor so values are visible
    try:
        # Update any live labels by touching the getters once
        _ = (_get_optical_nm(), _get_motor_nm())
    except Exception:
        pass
    try:
        show_lambda_window()
    except Exception:
        pass

# Connection
ttk.Label(left, text="Port").grid(row=0, column=0, sticky="w")
port_var = tk.StringVar(value=DEFAULT_PORT)
entry_port = ttk.Entry(left, textvariable=port_var, width=12); entry_port.grid(row=0, column=1, sticky="ew")

ttk.Label(left, text="Baud").grid(row=1, column=0, sticky="w")
baud_var = tk.StringVar(value=str(DEFAULT_BAUD))
entry_baud = ttk.Entry(left, textvariable=baud_var, width=12); entry_baud.grid(row=1, column=1, sticky="ew")

btn_connect = ttk.Button(left, text="Connect", command=on_connect); btn_connect.grid(row=2, column=0, sticky="ew", pady=(6,0))
btn_disconnect = ttk.Button(left, text="Disconnect", command=on_disconnect); btn_disconnect.grid(row=2, column=1, sticky="ew", pady=(6,0))

# Go To
ttk.Label(left, text="Go To (nm)").grid(row=3, column=0, sticky="w", pady=(12,0))
entry_goto = ttk.Entry(left, width=12); entry_goto.grid(row=3, column=1, sticky="ew", pady=(12,0))
btn_goto = ttk.Button(left, text="Go", command=goto_wavelength_action); btn_goto.grid(row=4, column=0, columnspan=2, sticky="ew")

# Scan params
ttk.Label(left, text="Start (nm)").grid(row=5, column=0, sticky="w", pady=(12,0))
entry_start = ttk.Entry(left, width=12); entry_start.grid(row=5, column=1, sticky="ew", pady=(12,0))

ttk.Label(left, text="End (nm)").grid(row=6, column=0, sticky="w")
entry_end = ttk.Entry(left, width=12); entry_end.grid(row=6, column=1, sticky="ew")

ttk.Label(left, text="Step (nm)").grid(row=7, column=0, sticky="w")
entry_step = ttk.Entry(left, width=12); entry_step.grid(row=7, column=1, sticky="ew")

ttk.Label(left, text="Wait (s)").grid(row=8, column=0, sticky="w")
entry_wait = ttk.Entry(left, width=12); entry_wait.grid(row=8, column=1, sticky="ew")


# Averaging controls
ttk.Label(left, text="Avg samples").grid(row=20, column=0, sticky="w")
avg_samples_var = tk.StringVar(value="8")
entry_avg_samples = ttk.Entry(left, textvariable=avg_samples_var, width=12); entry_avg_samples.grid(row=20, column=1, sticky="ew")

# Averaging mode: by Samples (N) or by Time (= Wait)
avg_mode_var = tk.StringVar(value="samples")  # 'samples' or 'time'
frm_avg_mode = ttk.Frame(left); frm_avg_mode.grid(row=21, column=0, columnspan=2, sticky="w")
ttk.Label(frm_avg_mode, text="Avg mode:").grid(row=0, column=0, sticky="w")
ttk.Radiobutton(frm_avg_mode, text="Samples", value="samples", variable=avg_mode_var).grid(row=0, column=1, sticky="w")
ttk.Radiobutton(frm_avg_mode, text="Time = Wait", value="time", variable=avg_mode_var).grid(row=0, column=2, sticky="w")

# Avg fraction (%) and Pre-settle (ms)
ttk.Label(left, text="Avg fraction (%)").grid(row=22, column=0, sticky="w")
avg_fraction_var = tk.StringVar(value="100")
entry_avg_fraction = ttk.Entry(left, textvariable=avg_fraction_var, width=12); entry_avg_fraction.grid(row=22, column=1, sticky="ew")

ttk.Label(left, text="Pre-settle (ms)").grid(row=23, column=0, sticky="w")
presettle_ms_var = tk.StringVar(value="0")
entry_presettle_ms = ttk.Entry(left, textvariable=presettle_ms_var, width=12); entry_presettle_ms.grid(row=23, column=1, sticky="ew")
ttk.Label(left, text="Current λ (nm)").grid(row=9, column=0, sticky="w")
entry_current = ttk.Entry(left, width=12); entry_current.grid(row=9, column=1, sticky="ew")
btn_set_current = ttk.Button(left, text="Set to Current Wavelength @ POS0", command=set_to_current_wavelength_action)
btn_set_current.grid(row=46, column=0, columnspan=2, sticky="ew", pady=(0,4))
entry_current.insert(0, f"{state.get('current_nm', 500.0):.3f}")

# --- Live Lambda Display (Optical vs Motor) ---
try:
    optical_lambda_var = tk.StringVar(value=f"{_get_optical_nm():.3f}")
    motor_lambda_var = tk.StringVar(value=f"{_get_motor_nm():.3f}")
    ttk.Label(left, text="Optical λ (nm)").grid(row=21, column=0, sticky="w")
    ttk.Label(left, textvariable=optical_lambda_var).grid(row=21, column=1, sticky="ew")
    ttk.Label(left, text="Motor λ (nm)").grid(row=22, column=0, sticky="w")
    ttk.Label(left, textvariable=motor_lambda_var).grid(row=22, column=1, sticky="ew")

    class _GUIBridge:
        def update_motor_lambda(self, nm):
            try:
                motor_lambda_var.set(f"{float(nm):.3f}")
            except Exception:
                pass
        def update_optical_lambda(self, nm):
            try:
                optical_lambda_var.set(f"{float(nm):.3f}")
            except Exception:
                pass

    globals()["gui"] = _GUIBridge()

    def show_lambda_window():
        try:
            win = tk.Toplevel(root)
            win.title("Lambda Monitor")
            ttk.Label(win, text="Optical λ (nm)").grid(row=0, column=0, sticky="w", padx=8, pady=6)
            ttk.Label(win, textvariable=optical_lambda_var).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
            ttk.Label(win, text="Motor λ (nm)").grid(row=1, column=0, sticky="w", padx=8, pady=6)
            ttk.Label(win, textvariable=motor_lambda_var).grid(row=1, column=1, sticky="ew", padx=8, pady=6)
            win.resizable(False, False)
        except Exception:
            pass
except Exception:
    pass



# Scan controls
btn_scan = ttk.Button(left, text="Start Scan", command=scan_action); btn_scan.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(8,0))
btn_pause = ttk.Button(left, text="Pause", command=pause_action); btn_pause.grid(row=11, column=0, sticky="ew")
btn_resume = ttk.Button(left, text="Resume", command=resume_action); btn_resume.grid(row=11, column=1, sticky="ew")
btn_stop = ttk.Button(left, text="Stop (Safe)", command=stop_action); btn_stop.grid(row=12, column=0, sticky="ew")
btn_hard = ttk.Button(left, text="Hard Kill (Reinit)", command=hard_kill_and_reinit); btn_hard.grid(row=12, column=1, sticky="ew", pady=(0,8))

# Jog controls
ttk.Label(left, text="Jog step (nm)").grid(row=13, column=0, sticky="w", pady=(8,0))
jog_step_var = tk.StringVar(value="0.5")
entry_jog_step = ttk.Entry(left, textvariable=jog_step_var, width=12); entry_jog_step.grid(row=13, column=1, sticky="ew", pady=(8,0))
btn_jog_minus = ttk.Button(left, text="Jog −", command=jog_minus_action); btn_jog_minus.grid(row=14, column=0, sticky="ew")
btn_jog_plus  = ttk.Button(left, text="Jog +", command=jog_plus_action);  btn_jog_plus.grid(row=14, column=1, sticky="ew")
btn_mark_point = ttk.Button(left, text="Mark Point", command=mark_point_action); btn_mark_point.grid(row=15, column=0, columnspan=2, sticky="ew")

# Multi-scan queue
ttk.Label(left, text="Scan Queue").grid(row=16, column=0, columnspan=2, sticky="w", pady=(12,0))
queue_listbox = tk.Listbox(left, height=5); queue_listbox.grid(row=17, column=0, columnspan=2, sticky="nsew")
btn_add_queue = ttk.Button(left, text="Add from fields", command=add_scan_to_queue); btn_add_queue.grid(row=18, column=0, sticky="ew", pady=(4,0))
btn_remove_queue = ttk.Button(left, text="Remove selected", command=remove_selected_scan); btn_remove_queue.grid(row=18, column=1, sticky="ew", pady=(4,0))
btn_run_queue = ttk.Button(left, text="Run queue", command=run_scan_queue_action); btn_run_queue.grid(row=19, column=0, sticky="ew")
btn_clear_queue = ttk.Button(left, text="Clear queue", command=clear_scan_queue); btn_clear_queue.grid(row=19, column=1, sticky="ew")


# Clear plot
btn_clear_plot = ttk.Button(left, text="Clear Plot", command=clear_plot_action); btn_clear_plot.grid(row=24, column=0, columnspan=2, sticky="ew", pady=(8,0))

from matplotlib.figure import Figure


# ========== Smart Preload (Δλ-based, pos/time-wait) BEGIN ==========
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional



# --- Calibration persistence helpers (non-destructive JSON update) ---
SETTINGS_PATH = "/mnt/data/monochromator_settings.json"

def _read_settings_json():
    try:
        import json, os
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _write_settings_json(data: dict):
    try:
        import json, os
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        try:
            log(f"[SETTINGS] Save failed: {e}", "warn")
        except Exception:
            pass
        return False

def _load_steps_per_nm_default():
    d = _read_settings_json()
    try:
        return int(d.get('steps_per_nm', 361765))
    except Exception:
        return 361765

def _save_steps_per_nm(val: int):
    d = _read_settings_json()
    d['steps_per_nm'] = int(val)
    _write_settings_json(d)
    try:
        log(f"[CAL] Saved steps_per_nm={int(val)}")
    except Exception:
        pass

# Patch _save_n_slip if it exists to merge keys;
# otherwise define a merged writer so we don't overwrite other fields.
try:
    _old__save_n_slip = _save_n_slip  # noqa: F821
except NameError:
    _old__save_n_slip = None

def _save_n_slip(val: int):
    d = _read_settings_json()
    d['n_slip_steps'] = int(val)
    _write_settings_json(d)
    try:
        log(f"[SLIP] Saved n_slip={int(val)} steps")
    except Exception:
        pass


Dir = int  # +1 forward, -1 backward

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

@dataclass
class SmartPreloader:
    # Required callbacks / parameters
    n_slip_steps: Dict[Dir, int]
    steps_per_nm_dir: Dict[Dir, float]
    read_opt_mech_nm: Callable[[], tuple]          # () -> (lam_opt_nm:float, lam_mech_nm:float)
    nm_to_steps: Callable[[float, Dir], int]       # (nm, dir) -> signed steps
    get_pos_steps: Callable[[], int]               # () -> encoder steps (int)
    send_lr_m: Callable[[int], None]               # (steps) -> send "LR{steps};M"
    wait_until_target: Callable[[int, float], bool]# (target_steps, timeout_s) -> bool
    sleep_fn: Callable[[float], None]              # pause-aware sleep preferred
    log_fn: Callable[[str], None]
    safe_stop_fn: Callable[[str], None]            # safe stop on timeout/error

    # Config with careful defaults
    sec_per_nm: float = 10.0                       # measured: 10 s per nm
    # timeout: add generous headroom to absorb controller latency + settling
    preload_timeout_min_s: float = 3.0
    preload_timeout_headroom: float = 1.6          # 60% extra time over estimate
    # settle time after preload (scales with move time)
    preload_settle_base_s: float = 0.03
    preload_settle_frac_of_move: float = 0.04      # 4% of move duration
    preload_settle_min_s: float = 0.03
    preload_settle_max_s: float = 0.80
    # tolerance band near target offset to classify "already loaded" vs "unloaded"
    tol_min_nm: float = 0.03
    tol_frac_of_slip: float = 0.10                 # 10% of |n_slip_nm|
    # expected sign of Δλ when properly loaded in each dir:
    # Δλ = λ_mech - λ_opt  ≈ sign_for_dir[dir] * n_slip_nm(dir)
    sign_for_dir: Dict[Dir, float] = field(default_factory=lambda: {+1: +1.0, -1: -1.0})
    # Enable/disable feature
    auto_preload_smart: bool = True

    # Internal references (optional, can be set via UI buttons)
    loaded_dir: Optional[Dir] = None
    ref_unloaded: Optional[Dict[str, float]] = None
    ref_loaded: Dict[Dir, Optional[Dict[str, float]]] = field(default_factory=lambda: {+1: None, -1: None})

    # Hysteresis memory (simple band stickiness)
    _last_class: Optional[str] = None

    def _n_slip_nm(self, dir_: Dir) -> float:
        return float(self.n_slip_steps[dir_]) / float(self.steps_per_nm_dir[dir_])

    def _expected_loaded_offset_nm(self, dir_: Dir) -> float:
        ref = self.ref_loaded.get(dir_)
        if ref:
            return (ref['lam_mech'] - ref['lam_opt'])
        return self.sign_for_dir[dir_] * self._n_slip_nm(dir_)

    def _classify(self, dir_: Dir, tol_enter_nm: float):
        lam_opt, lam_mech = self.read_opt_mech_nm()
        delta_now = lam_mech - lam_opt
        target_off = self._expected_loaded_offset_nm(dir_)
        if abs(delta_now - target_off) <= tol_enter_nm:
            klass = 'ALREADY_LOADED'
        elif abs(delta_now) <= tol_enter_nm:
            klass = 'UNLOADED'
        else:
            klass = 'NEEDS_PRELOAD'
        return klass, delta_now, target_off

    def _calc_preload_timing(self, slip_nm: float):
        move_time = abs(slip_nm) * self.sec_per_nm
        timeout_s = max(self.preload_timeout_min_s, move_time * (1.0 + self.preload_timeout_headroom))
        settle_s = self.preload_settle_base_s + self.preload_settle_frac_of_move * move_time
        settle_s = _clamp(settle_s, self.preload_settle_min_s, self.preload_settle_max_s)
        return timeout_s, settle_s, move_time

    def smart_preload_if_needed(self, dir_: Dir) -> bool:
        """
        Returns True if a preload was performed (and waited + settled).
        Returns False if already loaded in this dir or feature disabled.
        """
        if not self.auto_preload_smart:
            self.log_fn("[SMART] disabled; skipping smart preload check.")
            return False

        slip_nm = self._n_slip_nm(dir_)
        tol_enter_nm = max(self.tol_min_nm, self.tol_frac_of_slip * abs(slip_nm))
        klass, delta_now, target_off = self._classify(dir_, tol_enter_nm)

        # Simple hysteresis: if we were ALREADY_LOADED last time, keep it unless strongly violated
        if self._last_class == 'ALREADY_LOADED' and klass != 'ALREADY_LOADED':
            tol_sticky = tol_enter_nm * 1.2
            lam_opt2, lam_mech2 = self.read_opt_mech_nm()
            if abs((lam_mech2 - lam_opt2) - target_off) <= tol_sticky:
                klass = 'ALREADY_LOADED'

        self.log_fn(f"[SMART] dir={dir_:+d} Δλ_now={delta_now:.4f} nm, "
                    f"expect_off={target_off:.4f} nm, tol={tol_enter_nm:.4f} → {klass}")

        if klass == 'ALREADY_LOADED':
            self.loaded_dir = dir_
            self._last_class = klass
            return False

        # Compute timing before sending motion
        timeout_s, settle_s, move_time = self._calc_preload_timing(slip_nm)
        steps = self.nm_to_steps(abs(slip_nm), dir_)  # signed LR steps for this dir
        start_pos = self.get_pos_steps()
        target_steps = start_pos + steps

        self.log_fn(f"[SMART] Preload issuing LR{steps};M "
                    f"(~{abs(slip_nm):.3f} nm, est {move_time:.2f}s, timeout {timeout_s:.2f}s, settle {settle_s:.3f}s)")

        try:
            self.send_lr_m(steps)
        except Exception as e:
            self.safe_stop_fn(f"preload send failed: {e}")
            self._last_class = 'NEEDS_PRELOAD'
            self.loaded_dir = None
            return False

        ok = False
        try:
            ok = self.wait_until_target(target_steps, timeout_s)
        except Exception as e:
            self.log_fn(f"[SMART] wait_until_target raised: {e}")
            ok = False

        if not ok:
            self.safe_stop_fn("preload timeout/no-reach")
            self._last_class = 'NEEDS_PRELOAD'
            self.loaded_dir = None
            return False

        # Small settle after preload to let mechanics breathe
        self.sleep_fn(settle_s)
        self.loaded_dir = dir_
        self._last_class = 'ALREADY_LOADED'
        self.log_fn("[SMART] Preload complete and settled.")
        return True

    # --- UI helpers (can be bound to buttons) ---
    def set_unloaded_to_current(self):
        lam_opt, lam_mech = self.read_opt_mech_nm()
        self.ref_unloaded = {'lam_opt': lam_opt, 'lam_mech': lam_mech}
        self.loaded_dir = None
        self.log_fn(f"[SMART] Set Unloaded = Current (λ_opt={lam_opt:.4f}, λ_mech={lam_mech:.4f}).")

    def set_loaded_to_current(self, dir_: Dir):
        lam_opt, lam_mech = self.read_opt_mech_nm()
        self.ref_loaded[dir_] = {'lam_opt': lam_opt, 'lam_mech': lam_mech}
        self.loaded_dir = dir_
        self.log_fn(f"[SMART] Set Loaded({'+FWD' if dir_==+1 else '−BACK'}) = Current "
                    f"(Δλ={lam_mech - lam_opt:.4f} nm).")

    def get_loaded_label(self) -> str:
        if self.loaded_dir == +1:
            return "FWD"
        if self.loaded_dir == -1:
            return "BACK"
        return "NONE"
# ========== Smart Preload (Δλ-based, pos/time-wait) END ==========



fig = Figure(figsize=(5,4), dpi=100)
ax = fig.add_subplot(111)
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("PMT Voltage (V)")
line, = ax.plot([], [])
canvas = FigureCanvasTkAgg(fig, master=right)
canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
canvas.draw()
# Plot manager for multi-scan + jog
plot_mgr = PlotManager(root, ax, canvas)

# === MOVED: CSV helper + export functions placed before mainloop ===
# ---- CSV Export helpers ----







# Console
try:
    _console_parent = right
except NameError:
    _console_parent = left
console = scrolledtext.ScrolledText(_console_parent, height=10, state="disabled")
console.grid(row=2, column=0, sticky="ew")
console.tag_config("info", foreground="#111")
console.tag_config("warn", foreground="#b58900")
console.tag_config("error", foreground="#dc322f")

# Initial button states
set_buttons_connected(False)
set_fsm("IDLE")


def on_close():
    try:
        state["stop_flag"] = True
        if ser_ok():
            safe_stop()
            state["ser"].close()
    except Exception:
        print("[WARN] on_close: suppressed exception during cleanup")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)


# --- moved helpers (were after __main__) ---
def get_points(self, plot_id):
    """Return xs, ys for a scan series from the internal store."""
    try:
        # Primary: our _plots dict used by this PlotManager
        with self._lock:
            rec = self._plots.get(plot_id)
            if rec:
                xs = list(rec.get("x") or [])
                ys = list(rec.get("y") or [])
                return xs, ys
    except Exception:
        pass
    # Fallbacks for legacy structures (will be empty on this build)
    try:
        d = getattr(self, "_store", {}).get(plot_id)
        if d:
            xs = d.get("xs") or d.get("x") or []
            ys = d.get("ys") or d.get("y") or []
            return list(xs), list(ys)
    except Exception:
        pass
    try:
        rec = getattr(self, "series", {}).get(plot_id)
        if rec:
            return list(rec.get("xs", [])), list(rec.get("ys", []))
    except Exception:
        pass
    return [], []
try:
    PlotManager.get_points = get_points
except Exception:
    pass
# ---- CSV Export helpers ----
def _ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def _safe_filename(dirpath, fname):
    base, ext = os.path.splitext(fname)
    out = os.path.join(dirpath, fname)
    n = 1
    while os.path.exists(out):
        out = os.path.join(dirpath, f"{base}-{n:03d}{ext}")
        n += 1
    return out

def _build_filename(planned, pattern, scanid=None, queueid=None):
    now = datetime.datetime.now()
    tokens = dict(
        date=now.strftime("%Y%m%d"),
        time=now.strftime("%H%M%S"),
        start=f"{planned.get('s_nm', 0):.3f}",
        end=f"{planned.get('e_nm', 0):.3f}",
        step=f"{planned.get('st_nm', 0):.3f}",
        wait=f"{planned.get('dwell', 0):.3f}",
        mode=("time" if avg_mode_var.get()=="time" else "samples") if 'avg_mode_var' in globals() else "",
        samples=(avg_samples_var.get() if 'avg_samples_var' in globals() else ""),
        frac=(f"{_get_avg_fraction():.1f}" if '_get_avg_fraction' in globals() else ""),
        presettle=(f"{_get_presettle_sec()*1000:.0f}" if '_get_presettle_sec' in globals() else ""),
        scanid=scanid or "",
        queueid=queueid or "",
    )
    try:
        fname = pattern.format(**tokens)
    except Exception:
        fname = f"scan_{tokens['date']}_{tokens['time']}.csv"
    return "".join(c for c in fname if c not in r'\/\:*?"<>|')

def _scan_metadata(scanid):
    p = state.get('planned_params', {}) if 'state' in globals() else {}
    meta = {
        "scan_id": scanid,
        "started_at": state.get("scan_started_at") if 'state' in globals() else "",
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "status": ("interrupted" if (state.get("stop_flag") if 'state' in globals() else False) else "completed"),
        "start_nm": p.get("s_nm"), "end_nm": p.get("e_nm"),
        "step_nm": p.get("st_nm"), "wait_s": p.get("dwell"),
        "avg_mode": ("time" if (avg_mode_var.get()=="time" if 'avg_mode_var' in globals() else False) else "samples"),
        "avg_samples": (avg_samples_var.get() if 'avg_samples_var' in globals() else ""),
        "avg_fraction_pct": (_get_avg_fraction() if '_get_avg_fraction' in globals() else ""),
        "pre_settle_ms": (int(_get_presettle_sec()*1000) if '_get_presettle_sec' in globals() else ""),
    }
    return meta

def export_scan_csv_by_id(plot_id, planned, folder, pattern, scanid=None, queueid=None):
    try:
        xs, ys = plot_mgr.get_points(plot_id)
    except Exception:
        xs, ys = [], []
    _ensure_dir(folder)
    fname = _build_filename(planned or {}, pattern or "scan_{date}_{time}.csv", scanid, queueid)
    path = _safe_filename(folder, fname)
    meta = _scan_metadata(scanid)

    with open(path, "w", newline="", encoding="utf-8") as f:
        for k,v in meta.items():
            f.write(f"# {k}: {v}\n")
        f.write("# columns: index,wavelength_nm,voltage_V\n")
        w = csv.writer(f)
        w.writerow(["index","wavelength_nm","voltage_V"])
        for i, (x,y) in enumerate(zip(xs, ys)):
            try:
                w.writerow([i, f"{float(x):.6f}", f"{float(y):.6f}"])
            except Exception:
                w.writerow([i, x, y])
    try:
        log(f"[CSV] Saved {path}")
    except Exception:
        pass
    return path

def export_active_scan_csv():
    try:
        pid = state.get('active_plot_id')
    except Exception:
        pid = None
    if not pid:
        try:
            messagebox.showwarning("Export", "No active scan to export.")
        except Exception:
            pass
        return
    p = state.get('planned_params', {}) if 'state' in globals() else {}
    scanid = state.get("scan_id") if 'state' in globals() else None
    try:
        saved = export_scan_csv_by_id(pid, p, export_dir_var.get(), pattern_var.get(), scanid, state.get("queue_id") if 'state' in globals() else None)
    except Exception as e:
        try:
            messagebox.showerror("Export failed", str(e))
        except Exception:
            pass
        return
    return saved
# --- end moved helpers ---


def export_all_scans_and_summary():
    """
    Export every scan in the current PlotManager to CSV and write a queue_summary.csv.
    Uses export_dir_var for folder and export_scan_csv_by_id for each plot.
    """
    try:
        folder = export_dir_var.get() if 'export_dir_var' in globals() else os.path.expanduser("~/scans")
    except Exception:
        folder = os.path.expanduser("~/scans")
    try:
        _ensure_dir(folder)
    except Exception:
        pass
    rows = []
    try:
        qid = state.get("queue_id") if 'state' in globals() else ""
    except Exception:
        qid = ""
    try:
        # Gather all plot IDs from PlotManager
        if 'plot_mgr' in globals():
            with plot_mgr._lock:
                plot_ids = list(plot_mgr._plots.keys())
        else:
            plot_ids = []
    except Exception:
        plot_ids = []
    for pid in plot_ids:
        try:
            planned = state.get("planned_params", {}) if 'state' in globals() else {}
        except Exception:
            planned = {}
        try:
            scanid = state.get("scan_id") if 'state' in globals() else None
        except Exception:
            scanid = None
        try:
            saved = export_scan_csv_by_id(pid, planned, folder, "scan_{date}_{time}.csv", scanid, qid)
            rows.append({"queue_id": qid, "scan_id": scanid or "", "filepath": saved})
        except Exception as e:
            try:
                log(f"[CSV] Export failed for {pid}: {e}", "error")
            except Exception:
                pass
    # Write simple summary CSV
    try:
        summ = os.path.join(folder, f"queue_{qid or 'summary'}.csv")
        with open(summ, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["queue_id","scan_id","filepath"])
            for r in rows:
                w.writerow([r.get("queue_id",""), r.get("scan_id",""), r.get("filepath","")])
        try:
            log(f"[CSV] Queue summary saved: {summ}")
        except Exception:
            pass
    except Exception as e:
        try:
            log(f"[ERROR] Writing summary failed: {e}", "error")
        except Exception:
            pass


# --- Calibration window (GUI) ---
def open_calibration_window():
    win = tk.Toplevel(root)
    win.title("Calibration")
    win.geometry("360x180")
    frm = ttk.Frame(win, padding=12)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="Steps per nm:").grid(row=0, column=0, sticky="w")
    spn_var = tk.IntVar(value=int(globals().get("STEPS_PER_NM", 361765)))
    ent_spn = ttk.Entry(frm, textvariable=spn_var, width=14)
    ent_spn.grid(row=0, column=1, sticky="w")

    ttk.Label(frm, text="n_slip (steps, backward preload):").grid(row=1, column=0, sticky="w")
    nslip_var = tk.IntVar(value=int(state.get("n_slip_steps", 0)))
    ent_nslip = ttk.Entry(frm, textvariable=nslip_var, width=14)
    ent_nslip.grid(row=1, column=1, sticky="w")

    status = ttk.Label(frm, text="", foreground="grey")
    status.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6,0))

    def save_cal():
        try:
            spn = int(ent_spn.get())
            nsl = int(ent_nslip.get())
        except Exception:
            status.config(text="Please enter integers.", foreground="red")
            return
        globals()['STEPS_PER_NM'] = spn
        state['n_slip_steps'] = nsl
        state['n_slip_steps_back'] = nsl
        try: _save_steps_per_nm(spn)
        except Exception: pass
        try: _save_n_slip(nsl)
        except Exception: pass
        status.config(text=f"Saved: steps_per_nm={spn}, n_slip={nsl} steps", foreground="green")
        try: log(f"[CAL] Applied steps_per_nm={spn}; n_slip={nsl}")
        except Exception: pass

    ttk.Button(frm, text="Save", command=save_cal).grid(row=2, column=0, columnspan=2, pady=(10,0))
    win.bind("<Escape>", lambda e: win.destroy())

if __name__ == "__main__":
    root.mainloop()




# ==== NM<->STEPS HELPERS (AUTO-PATCH) ===============================
def steps_per_nm(dir_sign: int) -> float:
    try:
        if not USE_DIR_AWARE_K:
            return float(K_STEPS_PER_NM)
    except NameError:
        pass
    try:
        return float(kf if dir_sign > 0 else kb)
    except Exception:
        return float(globals().get('K_STEPS_PER_NM', 361_765.0))

def nm_to_steps(delta_nm: float, dir_sign: int) -> int:
    return int(round(float(delta_nm) * steps_per_nm(dir_sign)))

def steps_to_nm(steps: int, dir_sign: int) -> float:
    return float(steps) / steps_per_nm(dir_sign)
# ==== NM<->STEPS HELPERS (AUTO-PATCH END) ===========================



# ==== COUNTED MOVE IMMEDIATE (AUTO-PATCH) ===========================
def do_counted_step_immediate(step_nm_signed: float, label: str = "COUNTED"):
    '''Send LR;M immediately for the requested nm (no deadband).'''
    import time
    try:
        if step_nm_signed == 0:
            return True
        dir_sign = 1 if step_nm_signed > 0 else -1
        steps = nm_to_steps(abs(step_nm_signed), dir_sign)
        try:
            drain_rx_quiet()
        except Exception:
            pass
        ser.write(f"LR {steps if dir_sign>0 else -steps}\r".encode())
        ser.write(b"M\r")
        ok = False
        try:
            ok = wait_until_position_reached(preload=False)
        except TypeError:
            ok = wait_until_position_reached()
        except Exception:
            ok = False
        if not ok:
            SEC_PER_NM = globals().get("S_PER_NM_DEFAULT", 10.0)
            time.sleep(abs(step_nm_signed) * SEC_PER_NM)
        if globals().get("SAFE_AFTER_MOVE", True):
            try:
                ser.write(b"ST\r"); ser.write(b"HP0\r"); ser.write(b"V0\r")
                drain_rx_quiet()
            except Exception:
                pass
        old_m = state['motor_lambda_nm']; old_o = state['optical_lambda_nm']
        state['motor_lambda_nm']  = old_m + step_nm_signed
        state['optical_lambda_nm'] = old_o + step_nm_signed
        try:
            gui.update_motor_lambda(state['motor_lambda_nm']); gui.update_optical_lambda(state['optical_lambda_nm'])
        except Exception:
            pass
        try:
            log(f"[{label}] Δλ={step_nm_signed:+.6f} nm | motor {old_m:.6f}->{state['motor_lambda_nm']:.6f} | optical {old_o:.6f}->{state['optical_lambda_nm']:.6f}")
        except Exception:
            pass
        return True
    except Exception as e:
        try:
            log(f"[{label}] error: {e}")
        except Exception:
            pass
        return False
# ==== COUNTED MOVE IMMEDIATE (AUTO-PATCH END) =======================

