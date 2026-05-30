"""apex-agents-bench CLI.

Commands:
  apex-agents-bench info       -- environment + repo layout sanity check
  apex-agents-bench worlds     -- list the 33 worlds
  apex-agents-bench tasks      -- browse / filter the 480 tasks
  apex-agents-bench show       -- print one task in full
  apex-agents-bench models     -- list available agent profiles
  apex-agents-bench catalog    -- characterize the dataset, write JSON
  apex-agents-bench smoke      -- single-task end-to-end smoke run
  apex-agents-bench run        -- multi-task production runner
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from apex_agents_bench import __version__
from apex_agents_bench.config import (
    AGENT_MAX_STEPS,
    AGENT_TIMEOUT_SECONDS,
    DEFAULT_HOST_PORT,
    DEFAULT_JUDGE_MODEL,
    MCP_SERVERS,
    RUNS_PER_TASK,
    VALID_DOMAINS,
    AgentRunConfig,
    JudgeConfig,
    Settings,
)
from apex_agents_bench.paths import (
    archipelago_agents_dir,
    archipelago_environment_dir,
    archipelago_grading_dir,
    default_dataset_dir,
    repo_root,
    runs_dir,
    vendor_dir,
)

app = typer.Typer(
    name="apex-agents-bench",
    help="Reproducible runner around the Mercor Archipelago harness.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _validate_domain(domain: str | None) -> None:
    """Reject an unknown --domain with a clear error listing valid options.

    Domain strings in the apex-agents dataset are case-sensitive and include
    spaces (e.g. "Investment Banking"); the literal placeholder filter
    semantics would silently return zero rows on a near-miss like "banking".
    """
    if domain is None or domain in VALID_DOMAINS:
        return
    console.print(
        f"[red]error:[/red] --domain must be one of "
        f"{[repr(d) for d in VALID_DOMAINS]}; got {domain!r}.\n"
        'Quote multi-word domains, e.g. --domain "Investment Banking".'
    )
    raise typer.Exit(code=2)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


# -----------------------------------------------------------------------------


@app.callback()
def _main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logs."),
) -> None:
    _setup_logging(verbose)


@app.command()
def version() -> None:
    """Print the apex-agents-bench package version."""
    console.print(__version__)


# -----------------------------------------------------------------------------


@app.command()
def info() -> None:
    """Show repo layout, resolved paths, and a Docker / HF token probe."""
    table = Table(show_header=False, box=None)
    table.add_row("[bold]apex-agents-bench version[/bold]", __version__)
    table.add_row("[bold]repo root[/bold]", str(repo_root()))
    table.add_row("[bold]vendor[/bold]", str(vendor_dir()))
    table.add_row("[bold]agents pkg[/bold]", str(archipelago_agents_dir()))
    table.add_row("[bold]grading pkg[/bold]", str(archipelago_grading_dir()))
    table.add_row("[bold]environment pkg[/bold]", str(archipelago_environment_dir()))
    table.add_row("[bold]dataset dir (default)[/bold]", str(default_dataset_dir()))
    table.add_row("[bold]runs dir[/bold]", str(runs_dir()))
    table.add_row("[bold]default judge[/bold]", DEFAULT_JUDGE_MODEL)
    table.add_row(
        "[bold]agent caps (policy)[/bold]",
        f"max_steps={AGENT_MAX_STEPS}, timeout={AGENT_TIMEOUT_SECONDS}s",
    )
    table.add_row("[bold]runs per task (policy)[/bold]", str(RUNS_PER_TASK))
    table.add_row(f"[bold]MCP servers ({len(MCP_SERVERS)})[/bold]", ", ".join(MCP_SERVERS))
    console.print(table)
    console.print()

    # --- Probes -------------------------------------------------------------
    from apex_agents_bench.docker_env import docker_available

    if docker_available():
        console.print("[green]✓[/green] Docker daemon reachable")
    else:
        console.print("[yellow]![/yellow] Docker daemon not reachable -- `make docker-check`")

    if shutil.which("uv") is not None:
        console.print("[green]✓[/green] `uv` available on PATH")
    else:
        console.print(
            "[yellow]![/yellow] `uv` not on PATH -- "
            "install with `pip install uv` or see vendor/archipelago README."
        )

    if os.environ.get("HF_TOKEN"):
        console.print("[green]✓[/green] HF_TOKEN present")
    else:
        console.print(
            "[yellow]![/yellow] HF_TOKEN not set -- dataset fetch will fail until you set it."
        )


# -----------------------------------------------------------------------------


@app.command()
def models() -> None:
    """List available agent profiles. The judge is fixed by policy."""
    from apex_agents_bench.agent_profile import profiles_by_family

    console.print(
        f"[bold]Judge (fixed):[/bold] {DEFAULT_JUDGE_MODEL}  "
        "(OpenAI default reasoning_effort=medium)"
    )
    console.print()
    table = Table(title="Agent profiles", show_lines=False)
    table.add_column("profile name", style="bold", no_wrap=True)
    table.add_column("provider", no_wrap=True)
    table.add_column("orchestrator model", no_wrap=True)
    table.add_column("extra args", overflow="fold")
    table.add_column("notes", overflow="fold")
    for _family, ps in profiles_by_family().items():
        for p in ps:
            table.add_row(
                p.name,
                p.provider,
                p.orchestrator_model,
                ", ".join(f"{k}={v!r}" for k, v in p.orchestrator_extra_args.items()),
                p.notes,
            )
        table.add_section()
    console.print(table)


# -----------------------------------------------------------------------------


@app.command()
def worlds(
    input_dir: Path = typer.Option(
        default_dataset_dir(),
        "--input-dir",
        "-i",
        help="Path to the apex-agents dataset index (see `make fetch-dataset`).",
        resolve_path=True,
    ),
    domain: str | None = typer.Option(
        None,
        "--domain",
        "-d",
        help=f"Filter to one domain. One of: {', '.join(repr(d) for d in VALID_DOMAINS)}.",
    ),
) -> None:
    """List the 33 worlds in the APEX-Agents dataset."""
    from apex_agents_bench.dataset import DatasetError, load_tasks, load_worlds

    _validate_domain(domain)

    try:
        all_worlds = load_worlds(input_dir)
        all_tasks = load_tasks(input_dir)
    except DatasetError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=2) from e

    if domain:
        all_worlds = [w for w in all_worlds if w.domain == domain]

    tasks_per_world: dict[str, int] = {}
    for t in all_tasks:
        tasks_per_world[t.world_id] = tasks_per_world.get(t.world_id, 0) + 1

    table = Table(title=f"APEX-Agents worlds ({len(all_worlds)} shown)", show_lines=False)
    table.add_column("world_id", style="bold", no_wrap=True)
    table.add_column("domain", no_wrap=True)
    table.add_column("n tasks", justify="right")
    table.add_column("world_name", overflow="fold")
    for w in all_worlds:
        table.add_row(w.world_id, w.domain, str(tasks_per_world.get(w.world_id, 0)), w.world_name)
    console.print(table)


# -----------------------------------------------------------------------------


@app.command()
def tasks(
    input_dir: Path = typer.Option(
        default_dataset_dir(),
        "--input-dir",
        "-i",
        resolve_path=True,
    ),
    domain: list[str] | None = typer.Option(
        None,
        "--domain",
        "-d",
        help=f"Filter to one or more domains (repeatable). One of: "
        f"{', '.join(repr(d) for d in VALID_DOMAINS)}.",
    ),
    world: str | None = typer.Option(None, "--world", "-w", help="Filter to one world_id."),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Show first N after filtering."),
    full_prompt: bool = typer.Option(False, "--full-prompt", help="Print full prompts."),
) -> None:
    """Browse the 480 APEX-Agents tasks. Read-only; no API calls."""
    from apex_agents_bench.dataset import DatasetError, load_tasks
    from apex_agents_bench.task_index import build_index

    # Validate each domain string before any IO.
    for d in domain or ():
        _validate_domain(d)

    try:
        summaries = build_index(input_dir)
    except DatasetError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=2) from e

    if domain:
        summaries = [s for s in summaries if s.domain in set(domain)]
    if world:
        summaries = [s for s in summaries if s.world_id == world]
    if limit is not None:
        summaries = summaries[:limit]

    if not summaries:
        console.print("[yellow]no tasks match the filter[/yellow]")
        return

    if full_prompt:
        tasks_by_id = {t.task_id: t for t in load_tasks(input_dir)}
        for s in summaries:
            console.print()
            console.rule(f"[bold]{s.domain} · {s.world_id} · {s.task_id}[/bold]")
            console.print(
                f"task_name={s.task_name!r}  prompt_chars={s.prompt_chars}  "
                f"criteria={s.n_criteria}  has_input_files={s.has_input_files}"
            )
            console.print()
            console.print(tasks_by_id[s.task_id].prompt)
        return

    table = Table(title=f"APEX-Agents tasks ({len(summaries)} shown)", show_lines=False)
    table.add_column("task_id", style="bold", no_wrap=True)
    table.add_column("domain", no_wrap=True)
    table.add_column("world", no_wrap=True)
    table.add_column("crit", justify="right")
    table.add_column("inp", justify="center")
    table.add_column("first sentence", overflow="fold")
    for s in summaries:
        table.add_row(
            s.task_id,
            s.domain,
            s.world_id,
            str(s.n_criteria),
            "*" if s.has_input_files else "-",
            s.first_sentence,
        )
    console.print(table)


# -----------------------------------------------------------------------------


@app.command()
def show(
    task_id: str = typer.Argument(..., help="Task ID to print in full."),
    input_dir: Path = typer.Option(
        default_dataset_dir(),
        "--input-dir",
        "-i",
        resolve_path=True,
    ),
) -> None:
    """Print one task in full: prompt + rubric."""
    from apex_agents_bench.dataset import DatasetError, get_task

    try:
        t = get_task(input_dir, task_id)
    except DatasetError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=2) from e

    console.rule(f"[bold]{t.domain} · {t.world_id} · {t.task_id}[/bold]")
    console.print(
        f"task_name={t.task_name!r}  prompt_chars={t.prompt_chars}  "
        f"criteria={t.n_criteria}  has_input_files={t.task_input_files}"
    )
    console.print()
    console.print("[bold]PROMPT[/bold]")
    console.print(t.prompt)
    console.print()
    console.print("[bold]RUBRIC CRITERIA[/bold]")
    if not t.rubric:
        console.print("  (none)")
    else:
        for i, c in enumerate(t.rubric):
            marker = "* " if i == 0 else "  "
            console.print(f"{marker}[bold]{c.verifier_id}[/bold]  {c.criteria}")
        console.print()
        console.print("[dim]* = primary objective (highest-weighted criterion)[/dim]")

    # --- Reference fields (shipped by the dataset; NOT consumed by grading) ---
    if t.gold_response_type or t.gold_response or t.expected_output:
        console.print()
        console.print(
            "[bold]REFERENCE METADATA[/bold]  "
            "[dim](NOT used by the published `output_llm` grading path -- "
            "shown for analysis only)[/dim]"
        )
        if t.gold_response_type:
            console.print(f"  gold_response_type: {t.gold_response_type!r}")
        if t.expected_output:
            preview = (
                t.expected_output
                if len(t.expected_output) <= 280
                else (t.expected_output[:277] + "...")
            )
            console.print(f"  expected_output: {preview}")
        if t.gold_response:
            preview = (
                t.gold_response if len(t.gold_response) <= 280 else (t.gold_response[:277] + "...")
            )
            console.print(f"  gold_response: {preview}")


# -----------------------------------------------------------------------------


@app.command()
def catalog(
    input_dir: Path = typer.Option(
        default_dataset_dir(),
        "--input-dir",
        "-i",
        resolve_path=True,
    ),
    output: Path = typer.Option(
        Path("data") / "catalog.json",
        "--output",
        "-o",
        resolve_path=True,
    ),
    no_timestamp: bool = typer.Option(False, "--no-timestamp", help="Bytes-stable output."),
) -> None:
    """Characterize the dataset; produce a deterministic JSON snapshot."""
    from apex_agents_bench.catalog import build_report, write_report
    from apex_agents_bench.dataset import DatasetError

    try:
        report = build_report(input_dir, include_timestamp=not no_timestamp)
    except DatasetError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=2) from e

    write_report(report, output)

    table = Table(title="APEX-Agents catalog", show_header=False, box=None)
    table.add_row("[bold]dataset dir[/bold]", report.dataset_dir)
    table.add_row("[bold]total tasks[/bold]", str(report.total_tasks))
    table.add_row("[bold]total worlds[/bold]", str(report.total_worlds))
    table.add_row(
        "[bold]domains by task count[/bold]",
        ", ".join(f"{k}={v}" for k, v in report.domains_by_task_count.items()),
    )
    table.add_row(
        "[bold]domains by world count[/bold]",
        ", ".join(f"{k}={v}" for k, v in report.domains_by_world_count.items()),
    )
    table.add_row(
        "[bold]prompt chars (min/med/max)[/bold]",
        f"{report.prompt_chars.min} / {report.prompt_chars.median} / {report.prompt_chars.max}",
    )
    table.add_row(
        "[bold]criteria per task (min/med/max)[/bold]",
        f"{report.criteria_per_task.min} / {report.criteria_per_task.median} / {report.criteria_per_task.max}",
    )
    table.add_row("[bold]tasks with input files[/bold]", str(report.tasks_with_input_files))
    table.add_row("[bold]wrote[/bold]", str(output))
    console.print(table)


# -----------------------------------------------------------------------------


@app.command()
def smoke(
    model: str = typer.Option(..., "--model", "-m", help="Agent profile name."),
    judge_model: str = typer.Option(DEFAULT_JUDGE_MODEL, "--judge-model"),
    input_dir: Path = typer.Option(default_dataset_dir(), "--input-dir", "-i", resolve_path=True),
    domain: str | None = typer.Option(
        None,
        "--domain",
        help=f"Restrict the smoke task to one domain. One of: "
        f"{', '.join(repr(d) for d in VALID_DOMAINS)}.",
    ),
    world: str | None = typer.Option(
        None, "--world", help="Restrict the smoke task to one world_id."
    ),
    allow_input_files: bool = typer.Option(
        False,
        "--allow-input-files",
        help="If set, do not require the smoke task to be input-file-free.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Per-task output dir. Defaults to runs/smoke/<profile>__<task_id>/.",
        resolve_path=True,
    ),
    host_port: int = typer.Option(
        DEFAULT_HOST_PORT, "--host-port", help="Host port to expose the env on."
    ),
) -> None:
    """Run ONE task end-to-end. Verifies env-container + agent + grading."""
    from apex_agents_bench.agent_profile import get_profile
    from apex_agents_bench.smoke import render_result, run_smoke

    _validate_domain(domain)

    try:
        profile = get_profile(model)
    except KeyError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=2) from e

    settings = (
        Settings.defaults()
        .with_dataset_dir(input_dir)
        .with_judge(JudgeConfig(model_id=judge_model))
        .with_host_port(host_port)
    )

    try:
        result = run_smoke(
            settings,
            profile=profile,
            domain=domain,
            world_id=world,
            require_no_input_files=not allow_input_files,
            output_dir=output,
        )
    except Exception as e:
        console.print(f"[red]smoke failed:[/red] {e}")
        raise typer.Exit(code=1) from e

    console.print("[green]smoke OK[/green]")
    console.print(render_result(result))
    sys.exit(0)


# -----------------------------------------------------------------------------


@app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="Agent profile name."),
    domain: str | None = typer.Option(
        None,
        "--domain",
        "-d",
        help=f"Restrict to one domain. One of: {', '.join(repr(d) for d in VALID_DOMAINS)}.",
    ),
    world: str | None = typer.Option(None, "--world", "-w", help="Restrict to one world_id."),
    task_ids: str | None = typer.Option(
        None,
        "--task-ids",
        help="Comma-separated task ids (overrides --domain / --world / --limit).",
    ),
    start_index: int = typer.Option(
        0, "--start-index", min=0, help="Skip the first N after filters."
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="Run at most N tasks (after filters)."
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path. Defaults to runs/<timestamp>__<profile>__<scope>/results.csv.",
        resolve_path=True,
    ),
    input_dir: Path = typer.Option(default_dataset_dir(), "--input-dir", "-i", resolve_path=True),
    judge_model: str = typer.Option(DEFAULT_JUDGE_MODEL, "--judge-model"),
    host_port: int = typer.Option(
        DEFAULT_HOST_PORT, "--host-port", help="Host port to expose the env on."
    ),
    max_steps: int = typer.Option(
        AGENT_MAX_STEPS,
        "--max-steps",
        min=1,
        help=f"Agent step cap. Matches Mercor's published example default ({AGENT_MAX_STEPS}); "
        "raising it diverges from published numbers.",
    ),
    timeout_seconds: int = typer.Option(
        AGENT_TIMEOUT_SECONDS,
        "--timeout-seconds",
        min=60,
        help=f"Per-task wall-clock cap. Matches Mercor's published example default ({AGENT_TIMEOUT_SECONDS}s).",
    ),
    dc_rs: bool = typer.Option(
        False,
        "--dc-rs/--no-dc-rs",
        help="Enable the DC-RS (Dynamic Cheatsheet — Retrieval Synthesis, "
        "no-ground-truth) subsystem. Default: off. Faithful to Suzgun et "
        "al.: BEFORE each task a single synthesizer LLM call builds the "
        "per-domain cheatsheet from the previous cheatsheet plus the "
        "top-k retrieved past (task, trajectory) pairs; that cheatsheet "
        "is prepended to the user message of initial_messages.json. AFTER "
        "the agent runs, the just-completed (task, trajectory) is appended "
        "to the per-domain pool (no second LLM call). The synthesizer runs "
        "on the same model as the selected agent profile (only the judge "
        "is fixed at gpt-5.5). See docs/DC_RS_PRD.md.",
    ),
    dc_rs_top_k: int = typer.Option(
        3,
        "--dc-rs-top-k",
        min=0,
        help="Top-k retrieved past (task, trajectory) pairs the synthesizer "
        "sees per task when DC-RS is on. Suzgun's published value is 3.",
    ),
    trace: bool = typer.Option(
        False,
        "--trace/--no-trace",
        help="Enable the TRACE (uses-ground-truth) subsystem. "
        "Default: off. Mutually exclusive with --dc-rs. "
        "When on, each task is preceded by a dual-embedding retrieval "
        "into the per-domain cheatsheet; the cheatsheet block is "
        "prepended to the user message of initial_messages.json; the "
        "agent emits a <citations>[...] line on the last line of its "
        "final_answer.reasoning that is stripped before grading. After "
        "grading, the boolean criteria_passed==criteria_total is threaded "
        "into a reflector + curator pair (both same model as the agent), "
        "which produce <cheatsheet_updates> ops applied to the ledger. "
        "See docs/TRACE_PRD.md.",
    ),
    trace_top_k: int = typer.Option(
        8,
        "--trace-top-k",
        min=0,
        help="Top-k per retrieval axis when TRACE is on.",
    ),
    azure: bool = typer.Option(
        False,
        "--azure/--no-azure",
        help="Route GPT-5.5 chat completions (judge + agent profile + DC-RS "
        "synthesizer + TRACE reflector/curator) through Azure-OpenAI instead "
        "of OpenAI. Requires AZURE_API_KEY, AZURE_API_BASE, and "
        "AZURE_API_VERSION env vars; the Azure deployment name comes from "
        "AZURE_GPT55_DEPLOYMENT_NAME (default `gpt-5.5`). The embedding "
        "model (text-embedding-3-large) is always served by OpenAI "
        "regardless of this flag.",
    ),
) -> None:
    """Run APEX-Agents on a slice of tasks. ONE run per (task, model), always.

    Examples:
      apex-agents-bench run --model grok-4.3-high --domain "Investment Banking" --limit 5
      apex-agents-bench run --model gpt-5.5-medium --world <world_id>
      apex-agents-bench run --model gpt-5.5-medium --task-ids task_9ba58a6...,task_abc...
    """
    from apex_agents_bench.agent_profile import get_profile
    from apex_agents_bench.azure_routing import AzureConfig
    from apex_agents_bench.dc_rs.config import DCRSConfig
    from apex_agents_bench.runner import RunOptions
    from apex_agents_bench.runner import run as run_runner
    from apex_agents_bench.trace.config import TraceConfig

    if dc_rs and trace:
        console.print(
            "[red]error:[/red] --dc-rs and --trace are mutually exclusive; pick one."
        )
        raise typer.Exit(code=2)

    _validate_domain(domain)

    try:
        profile = get_profile(model)
    except KeyError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=2) from e

    ids_tuple: tuple[str, ...] | None = None
    if task_ids:
        ids_tuple = tuple(s.strip() for s in task_ids.split(",") if s.strip())
        if not ids_tuple:
            console.print("[red]error:[/red] --task-ids was empty after parsing.")
            raise typer.Exit(code=2)

    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        scope = domain or world or ("ids" if ids_tuple else "all")
        output = runs_dir() / f"{stamp}__{profile.name}__{scope}" / "results.csv"

    settings = (
        Settings.defaults()
        .with_dataset_dir(input_dir)
        .with_judge(JudgeConfig(model_id=judge_model))
        .with_agent(AgentRunConfig(max_steps=max_steps, timeout_seconds=timeout_seconds))
        .with_host_port(host_port)
    )

    opts = RunOptions(
        profile=profile,
        settings=settings,
        output_csv=output,
        output_dir=output.parent,
        domain=domain,
        world_id=world,
        task_ids=ids_tuple,
        start_index=start_index,
        limit=limit,
        dc_rs=DCRSConfig(
            enabled=dc_rs,
            top_k=dc_rs_top_k,
        ),
        trace=TraceConfig(
            enabled=trace,
            top_k_per_axis=trace_top_k,
        ),
        azure=AzureConfig(enabled=azure),
    )

    console.print(
        f"[bold]Starting run[/bold]  profile={profile.name}  "
        f"domain={domain or '(any)'}  world={world or '(any)'}  "
        f"limit={limit if limit is not None else 'no-cap'}  "
        f"judge={judge_model}\n  -> output: {output}"
    )
    try:
        stats = run_runner(opts)
    except KeyboardInterrupt:
        console.print(
            "[yellow]interrupted[/yellow] -- partial results saved in CSV; "
            "re-run with the same --output to resume."
        )
        raise typer.Exit(code=130) from None
    except Exception as e:
        console.print(f"[red]run failed:[/red] {e}")
        raise typer.Exit(code=1) from e

    table = Table(title="APEX-Agents run summary", show_header=False)
    table.add_row("[bold]profile[/bold]", profile.name)
    table.add_row("[bold]judge[/bold]", judge_model)
    table.add_row("[bold]tasks completed[/bold]", str(stats.get("total_completed", 0)))
    table.add_row("[bold]overall mean[/bold]", f"{stats.get('overall_mean', 0.0):.4f}")
    for dom, info in (stats.get("by_domain") or {}).items():
        table.add_row(f"  [bold]{dom}[/bold] (n={info['n']})", f"{info['mean']:.4f}")
    table.add_row("[bold]CSV[/bold]", str(output))
    console.print(table)
