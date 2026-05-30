# Evaluation plan — apex-agents-bench

This document fixes the execution order for every (domain, world, method) cell on Mercor's APEX-Agents benchmark. The order is pre-registered so that rollout decisions cannot drift in response to interim results.

## Ordering rules

1. **Within a domain**, worlds are scheduled in **lexicographic ascending order of `world_id`**. The rule takes zero parameters; the order is recoverable from the dataset alone.
2. **Across domains**, the order is **lexicographic ascending by domain name** — Investment Banking, then Law, then Management Consulting. The rollout is **depth-first**: finish a domain's full world list before moving to the next domain. Per-domain state isolation means there is no cross-domain transfer to capture, so depth-first lets each within-domain cheatsheet/pool (DC-RS) or ledger (TRACE) reach steady state before the rollout moves on.
3. **Per (domain, world), the three methods** are run in this order: **baseline (no memory) → DC-RS → TRACE**. All three share the same task slice, the same test profile, and the same judge; relative order is fixed only for reproducibility.
4. **Resume**: each method has a single CSV per repo (one for baseline, one for DC-RS, one for TRACE) and the runner appends to it. The DC-RS per-domain pool + cheatsheet and the TRACE ledger carry forward across worlds within a domain via the per-domain state under `runs/<run>/dc_rs/<Domain>/` or `runs/<run>/trace/<Domain>/`.

## Status — Investment Banking

| # | `world_id` | tasks | baseline | DC-RS | TRACE |
|---:|---|---:|:---:|:---:|:---:|
| 1 | `world_1e4d4288e63f4a08851a3cc441eb3ccb` | 14 | ✓ | ✓ | ✓ |
| 2 | `world_43a921f91f0f4d2c85d8bd2774f9e681` | 9 | ✓ | ✓ | — |
| 3 | `world_5859ae30d8744ae782a778a39af37853` | 17 | — | — | — |
| 4 | `world_5970ed13783a463181bdf38337f0cad1` | 19 | — | — | — |
| 5 | `world_767c001731ba4316a35908dbb107cf85` | 17 | — | — | — |
| 6 | `world_7cabc3536d2d45f3aa32634046c85921` | 17 | — | — | — |
| 7 | `world_802bca9c604244748d866ba9dde7decf` | 19 | — | — | — |
| 8 | `world_bc99fdca9e3b4ab99233d4d1c3e8b153` | 18 | — | — | — |
| 9 | `world_e9f523e7a94f45e2bc7ff7b649943e33` | 14 | — | — | — |
| 10 | `world_f83f49b3776b4b5e870c36091f7e2b0b` | 16 | — | — | — |

## Status — Law

| # | `world_id` | tasks | baseline | DC-RS | TRACE |
|---:|---|---:|:---:|:---:|:---:|
| 1 | `world_06051b9b10c94c079db1bac3b70c4c4b` | 20 | ✓ | ✓ | — |
| 2 | `world_10631647211d4c2080c5774c0ac1224e` | 13 | — | — | — |
| 3 | `world_4c8dea260e674f37abc700d5ac09fff9` | 13 | — | — | — |
| 4 | `world_72e117e476674c6db7f16db331644d9f` | 12 | — | — | — |
| 5 | `world_848bb733fcc544a3b9ef5b0ea7ab67ae` | 11 | — | — | — |
| 6 | `world_85a3713cd2794fdfb56e92161325a00e` | 11 | — | — | — |
| 7 | `world_95fe2c7d53ae4120b830d30539506334` | 13 | — | — | — |
| 8 | `world_9797d81fa71c4dbfb192e89a0f2ac811` | 8 | — | — | — |
| 9 | `world_aa672f35da64403f81004c0223f26a01` | 12 | — | — | — |
| 10 | `world_ac4631be289645f2ae7db48b1bd442d0` | 16 | — | — | — |
| 11 | `world_e81842899beb4631b2e07feafb4018dd` | 11 | — | — | — |
| 12 | `world_eec3883ca3c54c41a62d3f220a27736c` | 20 | — | — | — |

## Status — Management Consulting

| # | `world_id` | tasks | baseline | DC-RS | TRACE |
|---:|---|---:|:---:|:---:|:---:|
| 1 | `world_075ef4dff46146a580c8522e2ad29cb3` | 14 | ✓ | ✓ | — |
| 2 | `world_0f65ffc105a74cc79a207cbe7a2aff87` | 8 | — | — | — |
| 3 | `world_2a87e5cb5583475b820be279f6f46df6` | 18 | — | — | — |
| 4 | `world_2f84c98bb6ca4644937fa4f47b460c57` | 17 | — | — | — |
| 5 | `world_4120432b49c54a82bb938c46ad274f18` | 14 | — | — | — |
| 6 | `world_941eba667ba842f59662864b13b0554b` | 15 | — | — | — |
| 7 | `world_9b5ff332b34545a6aa211c5cab8a2dab` | 17 | — | — | — |
| 8 | `world_c0821d23e38342e9b9eeef5680a4fb69` | 15 | — | — | — |
| 9 | `world_d1b705c7393b40f9bb5e01bb63b99b91` | 15 | — | — | — |
| 10 | `world_d5110661c46c42a6bb952e6f6bd89967` | 16 | — | — | — |
| 11 | `world_d6c01a12c619445f8a9dda1973432337` | 11 | — | — | — |

## Continue the rollout

Each method has a single CSV per repo. Continuing extends that CSV; per-domain ledger snapshots carry forward.

**Progress so far (baseline + DC-RS):** the **first world of each domain** has been run — IB world 1 (`world_1e4d4288…`), MC world 1 (`world_075ef4df…`), and Law world 1 (`world_06051b9b…`) — plus IB world 2 (`world_43a921f9…`). TRACE has been run on IB world 1 only. Results are in [`results.md`](../results.md).

**Deviation note (recorded per the policy below).** The depth-first rule above would finish all of a domain's worlds before starting the next domain. In practice we instead ran the **first world of all three domains** first, to get an early cross-domain read on whether DC-RS generalizes beyond tool-heavy Investment Banking. Per-domain state isolation means this does not affect any within-domain result (each domain's cheatsheet/pool still starts empty at its first world). The remaining worlds revert to the depth-first order within each domain.

**Next pending worlds** (lexicographic within each domain):

```bash
# IB world 3
apex-agents-bench run --model grok-4.3-high \
    --world world_5859ae30d8744ae782a778a39af37853 \
    [--dc-rs] --output runs/grok43high-<method>/results.csv

# MC world 2
apex-agents-bench run --model grok-4.3-high \
    --world world_0f65ffc105a74cc79a207cbe7a2aff87 \
    [--dc-rs] --output runs/grok43high-<method>/results.csv

# Law world 2
apex-agents-bench run --model grok-4.3-high \
    --world world_10631647211d4c2080c5774c0ac1224e \
    [--dc-rs] --output runs/grok43high-<method>/results.csv
```

The per-domain DC-RS state trees (`dc_rs/Investment Banking/`, `dc_rs/Law/`, `dc_rs/Management Consulting/`) are isolated by construction — each domain's cheatsheet and pool start empty when that domain's first world runs, and both grow monotonically across that domain's worlds.

## Pre-registered, not post-hoc

This file is the source of truth for the rollout order. Any deviation from it must be noted alongside the corresponding run in [`results.md`](../results.md).
