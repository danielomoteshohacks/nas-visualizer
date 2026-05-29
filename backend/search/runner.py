import uuid
import time
import threading
import numpy as np
from typing import Callable, Optional

from backend.bench.search_space import get_arch_info, is_valid_arch
from backend.bench.proxies import score_architecture
from backend.bench.database import (
    init_db,
    upsert_proxy_scores,
    normalize_proxy_scores,
    get_top_by_proxy,
    get_top_by_accuracy,
    get_arch,
)
from backend.bench.trainer import train_architecture, get_cifar10_loaders
from backend.search.strategies import get_strategy


# -----------------------------------------------------------------------
# In-memory run registry
# Maps run_id (str) → SearchRun instance
# Every active and completed run lives here for the duration of the process
# -----------------------------------------------------------------------
_runs: dict = {}
_runs_lock  = threading.Lock()


class SearchRun:
    """
    One complete NAS experiment — from proxy scoring to final ranking.

    Lifecycle:
        idle       → created, not started
        scoring    → zero-cost proxies running
        training   → short training runs executing
        complete   → finished successfully
        error      → failed, self.error_message has details

    The run executes in a background thread so the FastAPI server
    stays responsive during the (potentially long) training phase.
    Progress events stream to the frontend via a registered callback.

    Every event emitted has this base shape:
        {
            'run_id':    str,
            'phase':     str,
            'timestamp': float,
            ...phase-specific fields...
        }
    """

    def __init__(self, config: dict):
        """
        config keys:
            strategy        'darts' | 'evolutionary' | 'random'
            proxy_budget    int  — architectures to score with proxies
            training_budget int  — top candidates to actually train
            seed            int  — for reproducibility
        """
        self.run_id          = str(uuid.uuid4())[:8]
        self.config          = config
        self.status          = 'idle'
        self.error_message   = None
        self.results         = {}
        self.events          = []        # full event log
        self.started_at      = None
        self.finished_at     = None
        self._callback: Optional[Callable] = None
        self._lock           = threading.Lock()

        init_db()

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit(self, event: dict):
        """
        Record a progress event and forward it to the WebSocket callback.

        Thread-safe: uses a lock because the training loop runs in a
        background thread while the WebSocket handler reads events
        from the main thread.
        """
        event['run_id']    = self.run_id
        event['timestamp'] = round(time.time(), 3)

        with self._lock:
            self.events.append(event)

        if self._callback:
            try:
                self._callback(event)
            except Exception:
                # Never let a callback failure kill the search
                pass

    def set_callback(self, fn: Callable):
        self._callback = fn

    # ------------------------------------------------------------------
    # Main search pipeline
    # ------------------------------------------------------------------

    def run(self):
        """
        Execute the full three-phase NAS pipeline.

        Phase A — Zero-cost proxy scoring
            Score `proxy_budget` architectures in seconds using
            grad_norm + synflow + jacob_cov ensemble. No training.

        Phase B — Short training of top candidates
            Take the top `training_budget` architectures by proxy score.
            Train each for 20 epochs on CIFAR-10. Record learning curves.
            Use learning curve extrapolation to estimate 200-epoch accuracy.

        Phase C — Final ranking and result assembly
            Rank all trained architectures by real validation accuracy.
            Emit the final result event with the best architecture.
        """
        self.started_at = time.time()
        self.status     = 'scoring'

        try:
            proxy_budget    = int(self.config.get('proxy_budget', 500))
            training_budget = int(self.config.get('training_budget', 20))
            strategy_name   = self.config.get('strategy', 'darts')
            seed            = int(self.config.get('seed', 42))

            np.random.seed(seed)
            strategy = get_strategy(strategy_name, seed=seed)

            # ==============================================================
            # PHASE A — Zero-cost proxy scoring
            # ==============================================================

            self._emit({
                'phase':   'phase_start',
                'name':    'proxy_scoring',
                'message': (
                    f'Phase 1 of 3 — Scoring {proxy_budget} architectures '
                    f'using zero-cost proxies. No training required.'
                ),
                'total':   proxy_budget,
            })

            candidates    = strategy.suggest(proxy_budget)
            proxy_results = []

            for i, arch_idx in enumerate(candidates):

                scores = score_architecture(arch_idx)

                # Store with a temporary proxy_score before normalization
                scores['proxy_score'] = float(scores.get('grad_norm', 0.0))
                upsert_proxy_scores(scores)
                proxy_results.append(scores)

                # Stream a progress update every 25 architectures
                # (too frequent = WebSocket flood, too infrequent = laggy UI)
                if (i + 1) % 25 == 0 or (i + 1) == proxy_budget:
                    self._emit({
                        'phase':      'proxy_progress',
                        'done':       i + 1,
                        'total':      proxy_budget,
                        'pct':        round(100.0 * (i + 1) / proxy_budget, 1),
                        'arch_index': arch_idx,
                        'operations': scores['operations'],
                        'grad_norm':  round(float(scores.get('grad_norm', 0)), 4),
                        'synflow':    round(float(scores.get('synflow',   0)), 4),
                        'jacob_cov':  round(float(scores.get('jacob_cov', 0)), 4),
                    })

            # Normalize all three proxy scores to [0,1] and compute ensemble
            normalize_proxy_scores()

            # Re-fetch with normalized scores for strategy update
            scored_pool = get_top_by_proxy(limit=proxy_budget)
            strategy.update(scored_pool, source='proxy')

            top10_proxy = scored_pool[:10]

            self._emit({
                'phase':          'proxy_complete',
                'message':        (
                    f'Proxy scoring done. Top candidate: '
                    f'arch #{top10_proxy[0]["arch_index"]} '
                    f'(score {top10_proxy[0]["proxy_score"]:.4f})'
                ),
                'top_10':         [
                    {
                        'arch_index':  a['arch_index'],
                        'proxy_score': round(float(a['proxy_score']), 4),
                        'arch_string': a['arch_string'],
                        'operations':  a['operations']
                                       if isinstance(a['operations'], list)
                                       else [],
                    }
                    for a in top10_proxy
                ],
                'strategy_state': strategy.state(),
            })

            # ==============================================================
            # PHASE B — Short training of top candidates
            # ==============================================================

            self.status      = 'training'
            top_candidates   = scored_pool[:training_budget]

            self._emit({
                'phase':   'phase_start',
                'name':    'training',
                'message': (
                    f'Phase 2 of 3 — Training top {len(top_candidates)} '
                    f'architectures for 20 epochs each on CIFAR-10.'
                ),
                'total':   len(top_candidates),
            })

            # Download CIFAR-10 once (170 MB, cached after first run)
            self._emit({
                'phase':   'data_loading',
                'message': (
                    'Loading CIFAR-10 — downloading ~170 MB on first run, '
                    'instant on subsequent runs.'
                ),
            })

            train_loader, val_loader = get_cifar10_loaders()

            self._emit({
                'phase':   'data_ready',
                'message': 'CIFAR-10 ready. Training starts now.',
            })

            training_results = []

            for i, arch_record in enumerate(top_candidates):
                arch_idx = arch_record['arch_index']
                arch     = get_arch_info(arch_idx)
                ops      = arch['operations']

                self._emit({
                    'phase':       'arch_start',
                    'arch_index':  arch_idx,
                    'arch_number': i + 1,
                    'total':       len(top_candidates),
                    'operations':  ops,
                    'arch_string': arch['arch_string'],
                    'message':     (
                        f'Training arch {i+1}/{len(top_candidates)} '
                        f'(index {arch_idx})'
                    ),
                })

                # Epoch-level callback streams live loss/accuracy curves
                def make_epoch_cb(idx):
                    def cb(update):
                        self._emit({
                            'phase':      'epoch',
                            'arch_index': idx,
                            **update,
                        })
                    return cb

                result = train_architecture(
                    arch_index        = arch_idx,
                    operations        = ops,
                    train_loader      = train_loader,
                    val_loader        = val_loader,
                    progress_callback = make_epoch_cb(arch_idx),
                )
                training_results.append(result)

                self._emit({
                    'phase':                  'arch_done',
                    'arch_index':             arch_idx,
                    'arch_number':            i + 1,
                    'val_accuracy':           result['val_accuracy'],
                    'extrapolated_accuracy':  result['extrapolated_accuracy'],
                    'learning_curve':         result['learning_curve'],
                    'loss_curve':             result['loss_curve'],
                    'params':                 result['params'],
                    'flops':                  result['flops'],
                })

            # Update strategy weights with real training signal
            strategy.update(training_results, source='training')

            # ==============================================================
            # PHASE C — Final results
            # ==============================================================

            final_ranking = get_top_by_accuracy(limit=10)
            best          = final_ranking[0] if final_ranking else {}

            duration = round(time.time() - self.started_at, 1)

            self.results = {
                'best_arch_index':        best.get('arch_index'),
                'best_val_accuracy':      best.get('val_accuracy'),
                'best_extrapolated_acc':  best.get('val_accuracy'),
                'best_arch_string':       best.get('arch_string'),
                'top_architectures':      final_ranking,
                'proxy_budget':           proxy_budget,
                'training_budget':        len(top_candidates),
                'strategy':               strategy_name,
                'duration_seconds':       duration,
                'strategy_state':         strategy.state(),
            }

            self.status      = 'complete'
            self.finished_at = time.time()

            self._emit({
                'phase':   'complete',
                'message': (
                    f'Search complete in {duration}s. '
                    f'Best architecture: #{best.get("arch_index")} '
                    f'— {best.get("val_accuracy", 0):.2f}% val accuracy.'
                ),
                **self.results,
            })

        except Exception as exc:
            self.status        = 'error'
            self.error_message = str(exc)
            self._emit({
                'phase':   'error',
                'message': f'Search failed: {str(exc)}',
            })
            raise

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize run metadata for the REST API."""
        return {
            'run_id':       self.run_id,
            'status':       self.status,
            'config':       self.config,
            'started_at':   self.started_at,
            'finished_at':  self.finished_at,
            'results':      self.results,
            'error':        self.error_message,
            'event_count':  len(self.events),
        }


# -----------------------------------------------------------------------
# Public API used by main.py
# -----------------------------------------------------------------------

def create_run(config: dict) -> SearchRun:
    """Create and register a new search run. Does not start it."""
    run = SearchRun(config)
    with _runs_lock:
        _runs[run.run_id] = run
    return run


def start_run(run: SearchRun, callback: Callable = None) -> threading.Thread:
    """
    Launch a search run in a background daemon thread.

    daemon=True means the thread dies automatically if the main
    process exits — no zombie threads left behind.
    """
    if callback:
        run.set_callback(callback)
    thread = threading.Thread(target=run.run, daemon=True, name=f'run-{run.run_id}')
    thread.start()
    return thread


def get_run(run_id: str) -> Optional[SearchRun]:
    """Look up a run by ID. Returns None if not found."""
    with _runs_lock:
        return _runs.get(run_id)


def list_runs() -> list:
    """Return metadata for all runs, newest first."""
    with _runs_lock:
        runs = list(_runs.values())
    runs.sort(key=lambda r: r.started_at or 0, reverse=True)
    return [r.to_dict() for r in runs]