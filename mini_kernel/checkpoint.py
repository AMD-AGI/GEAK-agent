"""
Checkpoint Manager for mini-kernel Agent

Manages checkpoints to ensure progress is never lost.

Directory structure:
.mini-kernel/
├── kernel.toml              # Config
├── baseline.json            # Initial performance metrics
├── session.log              # Human-readable log
└── checkpoints/
    ├── cp-001/
    │   ├── files/           # Snapshot of modified files
    │   ├── metrics.json     # Performance at this point
    │   ├── summary.txt      # What was changed
    │   └── diff.patch       # Git-style diff from previous
    ├── cp-002/
    │   └── ...
    └── best/                # Symlink to best performing checkpoint
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime
import difflib


@dataclass
class CheckpointMetrics:
    """Performance metrics at a checkpoint."""
    latency_us: float
    speedup_vs_baseline: float
    speedup_vs_previous: float
    tests_passed: bool
    iteration: int
    strategy: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CheckpointMetrics':
        return cls(**d)


@dataclass
class Checkpoint:
    """A checkpoint record."""
    id: str
    path: Path
    metrics: CheckpointMetrics
    summary: str
    files_modified: List[str]


class CheckpointManager:
    """
    Manages checkpoints for the optimization session.
    
    Features:
    - Auto-save after each successful optimization
    - Track best checkpoint
    - Easy rollback to any checkpoint
    - Diff between checkpoints
    """
    
    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.checkpoint_dir = self.work_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints: List[Checkpoint] = []
        self.best_checkpoint: Optional[Checkpoint] = None
        self.baseline_metrics: Optional[Dict[str, Any]] = None
        self._checkpoint_counter = 0
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self):
        """Load existing checkpoints from disk."""
        for cp_dir in sorted(self.checkpoint_dir.glob("cp-*")):
            if cp_dir.is_dir():
                metrics_file = cp_dir / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics_data = json.load(f)
                    
                    metrics = CheckpointMetrics.from_dict(metrics_data)
                    summary_file = cp_dir / "summary.txt"
                    summary = summary_file.read_text() if summary_file.exists() else ""
                    
                    cp = Checkpoint(
                        id=cp_dir.name,
                        path=cp_dir,
                        metrics=metrics,
                        summary=summary,
                        files_modified=[],
                    )
                    self.checkpoints.append(cp)
                    
                    # Track best
                    if self.best_checkpoint is None or metrics.latency_us < self.best_checkpoint.metrics.latency_us:
                        self.best_checkpoint = cp
                    
                    # Update counter
                    try:
                        num = int(cp_dir.name.split("-")[1])
                        self._checkpoint_counter = max(self._checkpoint_counter, num)
                    except:
                        pass
    
    def save_baseline(self, metrics: Dict[str, Any]):
        """Save baseline metrics."""
        self.baseline_metrics = metrics
        baseline_file = self.work_dir / "baseline.json"
        with open(baseline_file, "w") as f:
            json.dump(metrics, f, indent=2)
    
    def create_checkpoint(self, 
                         latency_us: float,
                         strategy: str,
                         iteration: int,
                         tests_passed: bool,
                         modified_files: Dict[str, str],
                         summary: str) -> Checkpoint:
        """
        Create a new checkpoint.
        
        Args:
            latency_us: Current latency
            strategy: Strategy that produced this result
            iteration: Current iteration number
            tests_passed: Whether tests passed
            modified_files: Dict of filepath -> content
            summary: Human-readable summary of changes
        
        Returns:
            The created Checkpoint
        """
        self._checkpoint_counter += 1
        cp_id = f"cp-{self._checkpoint_counter:03d}"
        cp_path = self.checkpoint_dir / cp_id
        cp_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate speedups
        baseline_latency = self.baseline_metrics.get("latency_us", latency_us) if self.baseline_metrics else latency_us
        previous_latency = self.checkpoints[-1].metrics.latency_us if self.checkpoints else baseline_latency
        
        speedup_vs_baseline = baseline_latency / latency_us if latency_us > 0 else 1.0
        speedup_vs_previous = previous_latency / latency_us if latency_us > 0 else 1.0
        
        # Create metrics
        metrics = CheckpointMetrics(
            latency_us=latency_us,
            speedup_vs_baseline=speedup_vs_baseline,
            speedup_vs_previous=speedup_vs_previous,
            tests_passed=tests_passed,
            iteration=iteration,
            strategy=strategy,
            timestamp=datetime.now().isoformat(),
        )
        
        # Save metrics
        with open(cp_path / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        # Save summary
        (cp_path / "summary.txt").write_text(summary)
        
        # Save modified files
        files_dir = cp_path / "files"
        files_dir.mkdir(exist_ok=True)
        
        for filepath, content in modified_files.items():
            # Save with safe filename
            safe_name = filepath.replace("/", "_").replace("\\", "_")
            (files_dir / safe_name).write_text(content)
        
        # Generate diff from previous checkpoint
        if self.checkpoints:
            diff = self._generate_diff(self.checkpoints[-1], modified_files)
            (cp_path / "diff.patch").write_text(diff)
        
        # Create checkpoint object
        checkpoint = Checkpoint(
            id=cp_id,
            path=cp_path,
            metrics=metrics,
            summary=summary,
            files_modified=list(modified_files.keys()),
        )
        
        self.checkpoints.append(checkpoint)
        
        # Update best if this is better
        if self.best_checkpoint is None or latency_us < self.best_checkpoint.metrics.latency_us:
            self.best_checkpoint = checkpoint
            # Update best symlink
            best_link = self.checkpoint_dir / "best"
            if best_link.exists():
                best_link.unlink()
            best_link.symlink_to(cp_path.name)
        
        return checkpoint
    
    def _generate_diff(self, previous: Checkpoint, 
                      current_files: Dict[str, str]) -> str:
        """Generate diff between previous checkpoint and current files."""
        diff_lines = []
        
        for filepath, current_content in current_files.items():
            safe_name = filepath.replace("/", "_").replace("\\", "_")
            prev_file = previous.path / "files" / safe_name
            
            if prev_file.exists():
                prev_content = prev_file.read_text()
                diff = difflib.unified_diff(
                    prev_content.splitlines(keepends=True),
                    current_content.splitlines(keepends=True),
                    fromfile=f"a/{filepath}",
                    tofile=f"b/{filepath}",
                )
                diff_lines.extend(diff)
            else:
                diff_lines.append(f"+++ New file: {filepath}\n")
        
        return "".join(diff_lines)
    
    def rollback(self, checkpoint_id: str = None) -> Optional[Dict[str, str]]:
        """
        Rollback to a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to rollback to (default: previous)
        
        Returns:
            Dict of filepath -> content to restore, or None if failed
        """
        target = None
        
        if checkpoint_id is None:
            # Rollback to previous
            if len(self.checkpoints) >= 2:
                target = self.checkpoints[-2]
            elif self.checkpoints:
                target = self.checkpoints[-1]
        elif checkpoint_id == "best":
            target = self.best_checkpoint
        else:
            # Find by ID
            for cp in self.checkpoints:
                if cp.id == checkpoint_id:
                    target = cp
                    break
        
        if target is None:
            return None
        
        # Load files from checkpoint
        files = {}
        files_dir = target.path / "files"
        if files_dir.exists():
            for f in files_dir.iterdir():
                # Restore original filename
                original_name = f.name.replace("_", "/")
                files[original_name] = f.read_text()
        
        return files
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get checkpoint by ID."""
        for cp in self.checkpoints:
            if cp.id == checkpoint_id:
                return cp
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints with summary."""
        result = []
        for cp in self.checkpoints:
            is_best = cp.id == (self.best_checkpoint.id if self.best_checkpoint else None)
            result.append({
                "id": cp.id,
                "latency_us": cp.metrics.latency_us,
                "speedup": cp.metrics.speedup_vs_baseline,
                "strategy": cp.metrics.strategy,
                "timestamp": cp.metrics.timestamp,
                "is_best": is_best,
                "summary": cp.summary[:50] + "..." if len(cp.summary) > 50 else cp.summary,
            })
        return result
    
    def compare_checkpoints(self, cp1_id: str, cp2_id: str) -> Dict[str, Any]:
        """Compare two checkpoints."""
        cp1 = self.get_checkpoint(cp1_id)
        cp2 = self.get_checkpoint(cp2_id)
        
        if not cp1 or not cp2:
            return {"error": "Checkpoint not found"}
        
        return {
            "cp1": {
                "id": cp1.id,
                "latency_us": cp1.metrics.latency_us,
                "strategy": cp1.metrics.strategy,
            },
            "cp2": {
                "id": cp2.id,
                "latency_us": cp2.metrics.latency_us,
                "strategy": cp2.metrics.strategy,
            },
            "latency_diff_us": cp2.metrics.latency_us - cp1.metrics.latency_us,
            "speedup_diff": cp2.metrics.speedup_vs_baseline - cp1.metrics.speedup_vs_baseline,
        }
    
    def export_best(self, output_path: Path) -> bool:
        """Export best checkpoint files to output path."""
        if not self.best_checkpoint:
            return False
        
        files_dir = self.best_checkpoint.path / "files"
        if not files_dir.exists():
            return False
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for f in files_dir.iterdir():
            shutil.copy(f, output_path / f.name)
        
        # Also copy metrics and summary
        shutil.copy(
            self.best_checkpoint.path / "metrics.json",
            output_path / "metrics.json"
        )
        shutil.copy(
            self.best_checkpoint.path / "summary.txt",
            output_path / "summary.txt"
        )
        
        return True


