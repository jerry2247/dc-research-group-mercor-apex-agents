"""apex-agents-bench -- reproducible runner around the Mercor Archipelago harness.

This package provides a thin orchestration layer over the vendored Mercor
Archipelago harness at ``vendor/archipelago/``. The vendored code is treated
as third-party source and is not modified; all customization (judge model
selection, agent profile registry, per-task fresh container, CSV resume)
lives here.

Public surface:
    apex_agents_bench.config         -- typed run-config (judge, agent caps, defaults)
    apex_agents_bench.paths          -- repo-root and data-dir resolution
    apex_agents_bench.dataset        -- APEX-Agents task / rubric / world loader
    apex_agents_bench.world          -- world snapshot fetch + cache
    apex_agents_bench.docker_env     -- environment container orchestration
    apex_agents_bench.agent_profile  -- agent profile registry (gpt-5.5-*, grok-4.3-*, deepseek-v4-pro-max)
    apex_agents_bench.judge          -- grading-settings + verifier builder
    apex_agents_bench.trajectory     -- trajectory log parsing
    apex_agents_bench.smoke          -- single-task end-to-end runner
    apex_agents_bench.runner         -- multi-task driver with CSV resume
    apex_agents_bench.catalog        -- dataset characterization
    apex_agents_bench.task_index     -- browseable task index
    apex_agents_bench.cli            -- Typer CLI entry point (``apex-agents-bench``)
"""

from apex_agents_bench.config import Settings

__all__ = ["Settings", "__version__"]
__version__ = "0.1.0"
