# Docker

> **Read this if you're asking:** "Docker is acting up. What do I do?"

## Prerequisites

| Component | Minimum | Notes |
|---|---|---|
| Docker Desktop (macOS / Windows) | 4.20+ | Includes Docker Engine 24+ and Compose v2. |
| Docker Engine (Linux) | 24+ | `apt install docker-ce` or distro equivalent. |
| Docker Compose | v2 (subcommand `docker compose`) | We use the v2 form, not the legacy `docker-compose` binary. |
| Free disk | ~20 GB | Env image ~2 GB, world snapshots ~18.7 GB cumulative. |
| Free RAM | ~4 GB | The env container plus its 9 MCP server subprocesses. |
| Free port 8080 | yes (default) | Or set `APEX_AGENTS_HOST_PORT=<other>` in `.env`. |

`make docker-check` verifies the first five. Run it once per machine
during setup.

## How the env image gets built

The vendor's `vendor/archipelago/environment/Dockerfile` builds a
`debian:trixie-slim` image with Python 3.13, Node, Chromium,
LibreOffice (Calc / Writer / Impress for spreadsheet / document /
presentation handling), and the 9 MCP server packages. The first
`docker compose up --build` does the build (~3-7 minutes); subsequent
runs reuse the cached image (~5-10 seconds to start a container).

## Troubleshooting matrix

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop quit or engine stopped | Start Docker Desktop, then `make docker-check`. On Linux: `sudo systemctl start docker`. |
| `bind: address already in use` for port 8080 | Another local service on 8080 | `lsof -i :8080` to see who. Either stop them, or set `APEX_AGENTS_HOST_PORT=8090` in `.env`. |
| `no matching manifest for linux/arm64/v8` | M1/M2 Mac pulling x86-only image | Should not happen with our setup -- the Dockerfile uses `debian:trixie-slim` which has arm64 builds. If it does, file an issue. |
| `docker compose up` hangs at "Building environment" | First-time build; libreoffice + chromium are large layers | Be patient (~3-7 min). Subsequent runs are fast. |
| Container starts but `/health` times out at 180s | Slow startup of FastMCP gateway / MCP server init | Check `docker compose logs` in `vendor/archipelago/environment/`. Most often: a missing dep in the image rebuild. Try `docker system prune` then re-run. |
| Container exits immediately | Bug in vendor or in your environment .env | `docker compose logs` shows the error. Common: a broken `S3_*` env var (we leave them empty -- env file copies from `.env.example`). |
| Agent runs but no MCP tools listed | `/apps` config didn't get posted, or `cwd` paths inside container are wrong | Verify by `curl localhost:8080/mcp/` returns a non-empty MCP server list. The vendor's example expects the cwd `/app/mcp_servers/<server>` for each server -- if you've edited the MCP config, double-check those paths. |
| Out of disk | Many stale containers / leaked snapshots | `docker ps -aq | xargs docker rm -f` (force-removes all containers). `docker volume prune`. |
| `unauthorized: invalid token` from Docker Hub | Rare; happens if Docker Hub rate-limits anonymous pulls | Log in via `docker login` (free account). |
| Container DNS issues | Multi-container compose networking weirdness | Our runner talks to the env at `http://localhost:<port>`, not by container name. If you need to talk container-to-container, fix the upstream compose file, not our wrapper. |

## Cleanup

- `make clean` -- removes Python caches but does NOT touch Docker.
- `docker ps -a` then `docker rm -f <id>` -- per-container cleanup.
- `docker system prune` -- aggressive: removes stopped containers,
  unused images, build cache. Recovers ~10s of GB on a busy machine.
- `docker compose down -v` from inside `vendor/archipelago/environment/`
  -- the canonical "destroy this env" call. Our runner does it
  automatically on `atexit` and on SIGINT.

## Why per-task fresh containers (and not one persistent container)

World filesystem state, calendar entries, mail, chat history all
persist inside the container. If we reused a container across tasks,
task 2 would see task 1's edits. Mercor's reference example does
fresh containers per task; we follow suit.

The cost: ~5-10s of container startup overhead per task. At ~3 minutes
per task average, this is <5% overhead -- worth it for clean task
isolation.

If you really want one-container-per-many-tasks for *debugging*
purposes only, run the agent directly via `vendor/archipelago/agents/`
without our wrapper, using the published `examples/hugging_face_task/`
flow. Don't try to disable our per-task fresh-container behavior --
it's load-bearing for the resume + reproducibility story.

## Production Linux notes (for the curious)

We've designed everything to run on macOS Docker Desktop because
that's where most local research happens. The same setup runs on
Linux Docker Engine without changes. CI / cloud setups would skip the
Desktop layer and use `docker compose` directly.

There are no Mercor-published instructions for running APEX-Agents on
non-x86 hardware. Our pin to upstream commit `3f4a8234` inherits
their compatibility (which is `linux/amd64` and `linux/arm64` per the
base image `debian:trixie-slim`'s manifests).
