"""Base classes for disaggregated serving srun steps."""

import os
import subprocess
from pathlib import Path
from typing import List


class BaseSrun:
    """Base class for disaggregated serving srun steps."""

    def __init__(self, config: dict, global_config: dict, allocated_nodes: dict, log_dir: Path, dry_run: bool = False):
        self.config = config
        self.global_config = global_config
        self.allocated_nodes = allocated_nodes
        self.log_dir = log_dir
        self.dry_run = dry_run
        self.process = None

    def get_workspace(self) -> Path:
        """Get workspace directory (git repo root + closed/NVIDIA)."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--show-toplevel'],
                capture_output=True,
                text=True,
                check=True
            )
            git_root = Path(result.stdout.strip())
            return git_root / 'closed' / 'NVIDIA'
        except subprocess.CalledProcessError:
            # Fallback to current directory if not in git repo
            return Path.cwd()

    def get_container_image(self) -> str:
        """Get container image path."""
        return self.global_config['container_image']

    def get_storage_path(self) -> str:
        """Get storage mount path."""
        return self.global_config.get('storage_path',
                                      '/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone')

    def get_slurm_jobid(self) -> str:
        """Get SLURM job ID from global config or environment."""
        if 'slurm_jobid' in self.global_config:
            return str(self.global_config['slurm_jobid'])
        else:
            jobid = os.environ.get('SLURM_JOBID')
            if not jobid:
                if self.dry_run:
                    return 'DRYRUN'
                raise RuntimeError("SLURM_JOBID not found. Run inside salloc or specify in config.")
            return jobid

    def _format_command(self, cmd: List[str]) -> str:
        """Format command with indentation for readability."""
        if not cmd:
            return ""

        lines = [cmd[0]]  # Command name
        indent = "    "

        for arg in cmd[1:]:
            if arg.startswith('--'):
                lines.append(f"{indent}{arg}")
            elif arg.startswith('-'):
                lines.append(f"{indent}{arg}")
            elif '=' in arg and not arg.startswith('-'):
                # Make target like RUN_ARGS=...
                lines.append(f"{indent}{arg}")
            else:
                # Non-flag argument (e.g., 'make', 'run_llm_server')
                lines.append(f"{indent}{arg}")

        return " \\\n".join(lines)

    def run_command(self, cmd: List[str], description: str, background: bool = False, log_prefix: str = None, env: dict = None):
        """Execute command with optional dry-run.

        Args:
            cmd: Command to execute
            description: Human-readable description
            background: Whether to run in background
            log_prefix: If provided, redirect stdout/stderr to {log_prefix}.stdout and {log_prefix}.stderr
            env: Optional dict of additional environment variables to pass to subprocess
        """
        print(f"\n{'='*80}")
        print(f"[{description}]")
        print(f"{'='*80}")
        print(self._format_command(cmd))
        if env:
            print(f"[Custom env vars: {env}]")

        if self.dry_run:
            print("[DRY RUN] Command not executed")
            return None

        # Merge custom env with current environment
        subprocess_env = os.environ.copy()
        if env:
            subprocess_env.update(env)

        if background:
            print("[Background execution]")
            if log_prefix:
                stdout_file = open(f"{log_prefix}.stdout", 'w')
                stderr_file = open(f"{log_prefix}.stderr", 'w')
                self.process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file, env=subprocess_env)
                # Store file handles so they stay open
                self.process._stdout_file = stdout_file
                self.process._stderr_file = stderr_file
            else:
                self.process = subprocess.Popen(cmd, env=subprocess_env)
            return self.process
        else:
            result = subprocess.run(cmd, capture_output=False, env=subprocess_env)
            if result.returncode != 0:
                print(f"WARNING: Command exited with code {result.returncode}")
            return result

    def launch(self):
        """Launch this srun step. Must be implemented by subclasses."""
        raise NotImplementedError
