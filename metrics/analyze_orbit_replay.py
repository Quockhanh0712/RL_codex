import json
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median

import pandas as pd


def load_replay(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_obs(replay: dict, step_idx: int, player_view: int = 0) -> dict:
    return replay["steps"][step_idx][player_view].get("observation", {})


def get_planets(replay: dict, step_idx: int, player_view: int = 0) -> list:
    return get_obs(replay, step_idx, player_view).get("planets", [])


def get_fleets(replay: dict, step_idx: int, player_view: int = 0) -> list:
    return get_obs(replay, step_idx, player_view).get("fleets", [])


def owner_stats(replay: dict, step_idx: int) -> dict:
    """
    Return planets, production, and ships controlled by each owner.
    Owner -1 means neutral.
    """
    stats = defaultdict(lambda: {"planets": 0, "prod": 0.0, "ships": 0.0})

    for p in get_planets(replay, step_idx, 0):
        pid, owner, x, y, radius, ships, prod = p
        owner = int(owner)
        stats[owner]["planets"] += 1
        stats[owner]["prod"] += float(prod)
        stats[owner]["ships"] += float(ships)

    return dict(stats)


def fleet_stats(replay: dict, step_idx: int) -> dict:
    """
    Return active fleet count and ships in flight for each player.
    """
    stats = defaultdict(lambda: {"fleets": 0, "ships": 0.0})

    for f in get_fleets(replay, step_idx, 0):
        if len(f) < 7:
            continue

        fid, owner, x, y, angle, from_planet_id, ships = f

        if int(fid) < 0:
            continue

        owner = int(owner)
        stats[owner]["fleets"] += 1
        stats[owner]["ships"] += float(ships)

    return dict(stats)


def action_stats(replay: dict) -> dict:
    """
    Count action commands and total ships sent by each player.
    """
    n_players = len(replay["steps"][0])
    result = {}

    for player in range(n_players):
        actions = []

        for step_idx, step_row in enumerate(replay["steps"]):
            for action in step_row[player].get("action") or []:
                from_planet, angle, ships = action
                actions.append({
                    "step": step_idx,
                    "from_planet": int(from_planet),
                    "angle": float(angle),
                    "ships": float(ships),
                })

        ships_list = [a["ships"] for a in actions]

        result[player] = {
            "commands": len(actions),
            "turns_with_action": len(set(a["step"] for a in actions)),
            "total_ships_sent": sum(ships_list),
            "avg_ships_per_command": mean(ships_list) if ships_list else 0.0,
            "median_ships_per_command": median(ships_list) if ships_list else 0.0,
            "max_ships_command": max(ships_list) if ships_list else 0.0,
            "first_action_step": min((a["step"] for a in actions), default=None),
            "last_action_step": max((a["step"] for a in actions), default=None),
        }

    return result


def planet_info_map(replay: dict, step_idx: int) -> dict:
    result = {}

    for p in get_planets(replay, step_idx, 0):
        pid, owner, x, y, radius, ships, prod = p
        result[int(pid)] = {
            "owner": int(owner),
            "ships": float(ships),
            "prod": float(prod),
        }

    return result


def ownership_changes(replay: dict) -> tuple[list, Counter]:
    """
    Detect planet ownership changes between consecutive steps.
    """
    changes = []
    prev = planet_info_map(replay, 0)

    for step_idx in range(1, len(replay["steps"])):
        cur = planet_info_map(replay, step_idx)

        for pid in sorted(set(prev.keys()) & set(cur.keys())):
            if prev[pid]["owner"] != cur[pid]["owner"]:
                changes.append({
                    "step": step_idx,
                    "planet": pid,
                    "from": prev[pid]["owner"],
                    "to": cur[pid]["owner"],
                    "prod": cur[pid]["prod"],
                    "ships_after": cur[pid]["ships"],
                })

        prev = cur

    transitions = Counter((c["from"], c["to"]) for c in changes)
    return changes, transitions


def build_timeseries(replay: dict) -> pd.DataFrame:
    rows = []

    for step_idx in range(len(replay["steps"])):
        os = owner_stats(replay, step_idx)
        fs = fleet_stats(replay, step_idx)

        p0 = os.get(0, {"planets": 0, "prod": 0, "ships": 0})
        p1 = os.get(1, {"planets": 0, "prod": 0, "ships": 0})
        neu = os.get(-1, {"planets": 0, "prod": 0, "ships": 0})

        f0 = fs.get(0, {"fleets": 0, "ships": 0})
        f1 = fs.get(1, {"fleets": 0, "ships": 0})

        row = {
            "step": step_idx,
            "p0_planets": p0["planets"],
            "p1_planets": p1["planets"],
            "neutral_planets": neu["planets"],
            "p0_prod": p0["prod"],
            "p1_prod": p1["prod"],
            "neutral_prod": neu["prod"],
            "p0_planet_ships": p0["ships"],
            "p1_planet_ships": p1["ships"],
            "neutral_ships": neu["ships"],
            "p0_fleets": f0["fleets"],
            "p1_fleets": f1["fleets"],
            "p0_fleet_ships": f0["ships"],
            "p1_fleet_ships": f1["ships"],
        }

        row["prod_adv"] = row["p0_prod"] - row["p1_prod"]
        row["planet_adv"] = row["p0_planets"] - row["p1_planets"]
        row["total_ship_adv"] = (
            row["p0_planet_ships"] + row["p0_fleet_ships"]
            - row["p1_planet_ships"] - row["p1_fleet_ships"]
        )

        rows.append(row)

    return pd.DataFrame(rows)


def first_step_where(df: pd.DataFrame, condition_col: str, threshold: float) -> int | None:
    matched = df.loc[df[condition_col] >= threshold, "step"]

    if matched.empty:
        return None

    return int(matched.min())


def episode_rewards(replay: dict, gamma: float = 0.99) -> dict:
    """
    Compute undiscounted and discounted return from per-step rewards.
    Orbit Wars usually has sparse terminal reward.
    """
    n_players = len(replay["steps"][0])
    result = {}

    for player in range(n_players):
        rewards = [
            step_row[player].get("reward", 0)
            for step_row in replay["steps"]
        ]

        undiscounted = sum(rewards)
        discounted = sum((gamma ** t) * r for t, r in enumerate(rewards))

        result[player] = {
            "episode_return": undiscounted,
            "discounted_return": discounted,
            "terminal_reward": rewards[-1] if rewards else None,
        }

    return result


def analyze_replay(path: str | Path, main_player: int = 0, gamma: float = 0.99) -> dict:
    replay = load_replay(path)
    steps = replay["steps"]
    final_step = len(steps) - 1

    rewards = replay.get("rewards", [])
    statuses = replay.get("statuses", [])
    max_steps = replay.get("configuration", {}).get("episodeSteps", None)

    df = build_timeseries(replay)
    actions = action_stats(replay)
    changes, transitions = ownership_changes(replay)
    returns = episode_rewards(replay, gamma=gamma)

    final_owner = owner_stats(replay, final_step)
    final_fleet = fleet_stats(replay, final_step)

    opponent = 1 - main_player

    opponent_zero_rows = df.loc[df[f"p{opponent}_planets"] == 0, "step"]
    opponent_eliminated_step = (
        int(opponent_zero_rows.min())
        if not opponent_zero_rows.empty
        else None
    )

    summary = {
        "episode_id": replay.get("info", {}).get("EpisodeId"),
        "agents": replay.get("info", {}).get("TeamNames"),
        "seed": replay.get("info", {}).get("seed"),
        "statuses": statuses,
        "raw_rewards": rewards,
        "episode_length": len(steps),
        "max_episode_steps": max_steps,
        "ended_early": bool(max_steps is not None and len(steps) < max_steps),
        "main_player": main_player,
        "main_player_return": returns[main_player]["episode_return"],
        "main_player_discounted_return": returns[main_player]["discounted_return"],
        "main_player_win": rewards[main_player] > rewards[opponent],
        "opponent_eliminated_step": opponent_eliminated_step,
        "final_owner_stats": final_owner,
        "final_fleet_stats": final_fleet,
        "action_stats": actions,
        "ownership_change_count": len(changes),
        "ownership_transitions": dict(transitions),
        "neutral_captures_by_main": transitions.get((-1, main_player), 0),
        "neutral_captures_by_opponent": transitions.get((-1, opponent), 0),
        "pvp_captures_by_main": transitions.get((opponent, main_player), 0),
        "pvp_captures_by_opponent": transitions.get((main_player, opponent), 0),
        "avg_production_advantage": float(df["prod_adv"].mean()),
        "avg_planet_advantage": float(df["planet_adv"].mean()),
        "avg_total_ship_advantage": float(df["total_ship_adv"].mean()),
        "first_step_prod_lead": first_step_where(df, "prod_adv", 1),
        "first_step_prod_adv_10": first_step_where(df, "prod_adv", 10),
        "first_step_planet_adv_5": first_step_where(df, "planet_adv", 5),
    }

    return {
        "summary": summary,
        "timeseries": df,
        "ownership_changes": pd.DataFrame(changes),
    }


def print_report(result: dict) -> None:
    s = result["summary"]

    print("\n=== EPISODE RL METRICS ===")
    print(f"Episode ID: {s['episode_id']}")
    print(f"Agents: {s['agents']}")
    print(f"Seed: {s['seed']}")
    print(f"Rewards: {s['raw_rewards']}")
    print(f"Main player win: {s['main_player_win']}")
    print(f"Episode length: {s['episode_length']} / {s['max_episode_steps']}")
    print(f"Ended early: {s['ended_early']}")
    print(f"Opponent eliminated step: {s['opponent_eliminated_step']}")
    print(f"Main player return: {s['main_player_return']}")
    print(f"Main player discounted return: {s['main_player_discounted_return']:.4f}")

    print("\n=== FINAL MAP CONTROL ===")
    for owner, stats in sorted(s["final_owner_stats"].items()):
        label = "Neutral" if owner == -1 else f"Player {owner}"
        print(
            f"{label}: planets={stats['planets']}, "
            f"prod={stats['prod']}, ships={stats['ships']}"
        )

    print("\n=== ACTION STATS ===")
    for player, stats in s["action_stats"].items():
        print(
            f"Player {player}: commands={stats['commands']}, "
            f"turns_with_action={stats['turns_with_action']}, "
            f"ships_sent={stats['total_ships_sent']}, "
            f"avg_ships/action={stats['avg_ships_per_command']:.2f}, "
            f"first_action={stats['first_action_step']}, "
            f"last_action={stats['last_action_step']}"
        )

    print("\n=== OWNERSHIP CHANGES ===")
    print(f"Total ownership changes: {s['ownership_change_count']}")
    print(f"Transitions: {s['ownership_transitions']}")
    print(f"Neutral captures by main: {s['neutral_captures_by_main']}")
    print(f"Neutral captures by opponent: {s['neutral_captures_by_opponent']}")
    print(f"PvP captures by main: {s['pvp_captures_by_main']}")
    print(f"PvP captures by opponent: {s['pvp_captures_by_opponent']}")

    print("\n=== ADVANTAGE METRICS ===")
    print(f"Avg production advantage: {s['avg_production_advantage']:.2f}")
    print(f"Avg planet advantage: {s['avg_planet_advantage']:.2f}")
    print(f"Avg total ship advantage: {s['avg_total_ship_advantage']:.2f}")
    print(f"First step with production lead: {s['first_step_prod_lead']}")
    print(f"First step with production advantage >= 10: {s['first_step_prod_adv_10']}")
    print(f"First step with planet advantage >= 5: {s['first_step_planet_adv_5']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_path", type=str)
    parser.add_argument("--main-player", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--export-prefix", type=str, default=None)
    args = parser.parse_args()

    result = analyze_replay(
        args.replay_path,
        main_player=args.main_player,
        gamma=args.gamma,
    )

    print_report(result)

    if args.export_prefix:
        prefix = Path(args.export_prefix)
        result["timeseries"].to_csv(prefix.with_suffix(".timeseries.csv"), index=False)
        result["ownership_changes"].to_csv(prefix.with_suffix(".ownership_changes.csv"), index=False)
        pd.DataFrame([result["summary"]]).to_json(
            prefix.with_suffix(".summary.json"),
            orient="records",
            indent=2,
            force_ascii=False,
        )