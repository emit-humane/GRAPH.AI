"""D2 — Live Graph Updater (was S6, Dynamic Multigraph Update).

Adds each new ``TransactionEvent`` as a directed edge to the in-memory
``nx.MultiDiGraph``, recomputes a small set of local graph features for the
sender and receiver, detects whether the new edge closes a cycle, and looks
up community-level risk from the offline-fit ``community_profiles`` (when
present). Output: ``LiveGraphFeatureVector``.

Per the R14/R15 addendum (docs/rules_r14_r15_addendum.md "S6 processing
additions"), this updater also computes:
    * receiver_in_degree_unique_24h
    * receiver_inflow_amount_cv
    * sender_is_relay_node
    * sender_last_inflow_amount
    * sender_last_inflow_gap_seconds
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable

import math
import networkx as nx
import numpy as np
import pandas as pd

from .schemas import LiveGraphFeatureVector, TransactionEvent

logger = logging.getLogger(__name__)


def _to_ts(value) -> pd.Timestamp:
    """Normalise to pd.Timestamp.

    Using Timestamp (not datetime) preserves nanosecond precision from the
    historical multigraph and avoids the "Discarding nonzero nanoseconds"
    warning that ``.to_pydatetime()`` would emit on every read.
    """
    if isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value)


class LiveGraphUpdater:
    """In-memory MultiDiGraph mutator + feature extractor.

    Args:
        graph: Optional pre-loaded MultiDiGraph (e.g., from
            artifacts/transaction_multigraph.pkl). When None, starts empty.
        community_profiles: Optional DataFrame with at minimum
            ``account_id, community_id`` and any of ``community_size,
            community_density, community_risk_score, community_benford_chi2``.
        graph_features: Optional DataFrame with ``account_id, pagerank,
            betweenness, in_degree, out_degree, ...``. Used for baseline
            metrics not cheap to recompute live (PageRank, betweenness).
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph | None = None,
        community_profiles: pd.DataFrame | None = None,
        graph_features: pd.DataFrame | None = None,
    ) -> None:
        self.G: nx.MultiDiGraph = graph if graph is not None else nx.MultiDiGraph()
        self._community_idx: dict[str, dict] = {}
        if community_profiles is not None and not community_profiles.empty:
            # L2B emits one row per COMMUNITY with a ``member_accounts`` list.
            # Expand into a per-account lookup so __process_event__ can do
            # O(1) sender/receiver community lookups.
            if "member_accounts" in community_profiles.columns:
                for _, row in community_profiles.iterrows():
                    profile = row.to_dict()
                    members = profile.pop("member_accounts", None)
                    if members is None:
                        continue
                    for member in members:
                        self._community_idx[str(member)] = profile
            elif "account_id" in community_profiles.columns:
                # Allow a pre-flattened per-account frame too.
                for _, row in community_profiles.iterrows():
                    self._community_idx[row["account_id"]] = row.to_dict()
        self._gfeat_idx: dict[str, dict] = {}
        if graph_features is not None and not graph_features.empty:
            for _, row in graph_features.iterrows():
                self._gfeat_idx[row["account_id"]] = row.to_dict()

    # --------------------------- core processing --------------------------- #

    def process_event(self, event: TransactionEvent) -> LiveGraphFeatureVector:
        sender = event.sender_account
        receiver = event.receiver_account
        current_ts = _to_ts(event.timestamp)

        # 1) cycle detection — does the NEW edge close a cycle?
        edge_creates_cycle, cycle_length = self._detect_cycle_pre_add(sender, receiver)

        # 2) add the edge (and any missing nodes)
        self._ensure_node(sender)
        self._ensure_node(receiver)
        self.G.add_edge(
            sender,
            receiver,
            key=event.transaction_id,
            timestamp=current_ts,
            amount=float(event.amount),
            currency=event.currency,
            transaction_type=event.transaction_type,
            payment_channel=event.payment_channel,
            is_international=event.is_international,
            sender_country=event.sender_country,
            receiver_country=event.receiver_country,
        )

        # 3) 2-hop directed neighborhood (post-add)
        two_hop_nodes, two_hop_edges = self._two_hop_neighborhood(sender)

        # 4) degree-based features
        sender_in_deg = self.G.in_degree(sender)
        sender_out_deg = self.G.out_degree(sender)
        sender_in_unique = self._unique_predecessors(sender)
        sender_out_unique = self._unique_successors(sender)
        receiver_in_deg = self.G.in_degree(receiver)
        receiver_out_deg = self.G.out_degree(receiver)

        sender_fan_in_score = self._fan_score(sender_in_deg, sender_in_unique)
        sender_fan_out_score = self._fan_score(sender_out_deg, sender_out_unique)

        # 5) local cycle counts on a 2-/3-hop induced subgraph
        sender_2hop_cycle_count, sender_3hop_cycle_count = self._local_cycle_counts(sender)

        # 6) community lookups
        s_comm = self._community(sender)
        r_comm = self._community(receiver)
        sender_community_id = int(s_comm.get("community_id", -1))
        receiver_community_id = int(r_comm.get("community_id", -1))
        sender_community_size = int(s_comm.get("community_size", 0))
        sender_community_density = float(s_comm.get("community_density", 0.0))
        sender_community_risk_score = float(s_comm.get("community_risk_score", 0.0))
        sender_benford_chi2_community = float(s_comm.get("community_benford_chi2", 0.0))
        receiver_community_risk_score = float(r_comm.get("community_risk_score", 0.0))
        shared_community = (
            sender_community_id == receiver_community_id and sender_community_id >= 0
        )

        # 7) baseline graph features (PageRank/betweenness) — defaults if absent
        gfeat = self._gfeat_idx.get(sender, {})
        sender_pagerank = float(gfeat.get("pagerank", 0.0))
        sender_betweenness = float(gfeat.get("betweenness", 0.0))

        # --- R14 — fan-in aggregation features (receiver side, 24h window) -- #
        recv_in_edges_24h = self._in_edges_within(receiver, current_ts, WINDOW_24H)
        recv_in_degree_unique_24h = len({u for (u, _, _, _) in recv_in_edges_24h})
        inbound_amounts = [d["amount"] for (_, _, _, d) in recv_in_edges_24h]
        if len(inbound_amounts) >= 2:
            arr = np.asarray(inbound_amounts, dtype=float)
            recv_inflow_amount_cv = float(arr.std() / (arr.mean() + 1e-9))
        else:
            recv_inflow_amount_cv = 999.0

        # --- R15 — layering relay features (sender side, 7d + last inflow) -- #
        sender_in_unique_7d = self._in_unique_within(sender, current_ts, WINDOW_7D)
        sender_out_unique_7d = self._out_unique_within(sender, current_ts, WINDOW_7D)
        sender_is_relay = (
            1 <= sender_in_unique_7d <= 2 and 1 <= sender_out_unique_7d <= 2
        )

        # Most recent inflow to sender BEFORE the current event
        last_inflow = self._last_inflow_before(sender, current_ts, exclude_edge_key=event.transaction_id)
        if last_inflow is not None:
            sender_last_inflow_amount = float(last_inflow["amount"])
            sender_last_inflow_gap_seconds = (
                current_ts - _to_ts(last_inflow["timestamp"])
            ).total_seconds()
        else:
            sender_last_inflow_amount = 0.0
            sender_last_inflow_gap_seconds = float("inf")

        return LiveGraphFeatureVector(
            transaction_id=event.transaction_id,
            sender_account=sender,
            receiver_account=receiver,
            sender_in_degree=int(sender_in_deg),
            sender_out_degree=int(sender_out_deg),
            sender_in_degree_unique=int(sender_in_unique),
            sender_out_degree_unique=int(sender_out_unique),
            sender_2hop_cycle_count=int(sender_2hop_cycle_count),
            sender_3hop_cycle_count=int(sender_3hop_cycle_count),
            sender_fan_in_score=float(sender_fan_in_score),
            sender_fan_out_score=float(sender_fan_out_score),
            sender_pagerank=sender_pagerank,
            sender_betweenness=sender_betweenness,
            sender_community_id=sender_community_id,
            sender_community_size=sender_community_size,
            sender_community_density=sender_community_density,
            sender_community_risk_score=sender_community_risk_score,
            sender_benford_chi2_community=sender_benford_chi2_community,
            receiver_in_degree=int(receiver_in_deg),
            receiver_out_degree=int(receiver_out_deg),
            receiver_community_id=receiver_community_id,
            receiver_community_risk_score=receiver_community_risk_score,
            edge_creates_cycle=edge_creates_cycle,
            cycle_length=int(cycle_length),
            shared_community=bool(shared_community),
            two_hop_neighborhood=sorted(two_hop_nodes),
            two_hop_edge_list=two_hop_edges,
            receiver_in_degree_unique_24h=int(recv_in_degree_unique_24h),
            receiver_inflow_amount_cv=float(recv_inflow_amount_cv),
            sender_is_relay_node=bool(sender_is_relay),
            sender_last_inflow_amount=float(sender_last_inflow_amount),
            sender_last_inflow_gap_seconds=float(sender_last_inflow_gap_seconds),
        )

    # ----------------------------- helpers --------------------------------- #

    def _ensure_node(self, node_id: str) -> None:
        if not self.G.has_node(node_id):
            self.G.add_node(node_id)

    def _detect_cycle_pre_add(self, sender: str, receiver: str, max_hops: int = 3) -> tuple[bool, int]:
        """Does there exist a path receiver → ... → sender in the graph
        as-it-is (BEFORE adding the new edge)? Returns (creates_cycle, length).
        """
        if not self.G.has_node(receiver) or not self.G.has_node(sender):
            return False, 0
        # BFS from receiver looking for sender, bounded by max_hops
        frontier = {receiver}
        visited = {receiver}
        for depth in range(1, max_hops + 1):
            next_frontier: set[str] = set()
            for node in frontier:
                for succ in self.G.successors(node):
                    if succ == sender:
                        return True, depth + 1  # cycle includes the new edge → length = depth+1
                    if succ not in visited:
                        next_frontier.add(succ)
                        visited.add(succ)
            frontier = next_frontier
            if not frontier:
                break
        return False, 0

    def _two_hop_neighborhood(self, node: str) -> tuple[set[str], list[tuple]]:
        if not self.G.has_node(node):
            return set(), []
        first_hop = set(self.G.successors(node)) | set(self.G.predecessors(node))
        first_hop.discard(node)
        second_hop: set[str] = set()
        for n in first_hop:
            second_hop.update(self.G.successors(n))
            second_hop.update(self.G.predecessors(n))
        second_hop.discard(node)
        nodes = first_hop | second_hop

        # Collect edges among the two-hop set (cap to avoid runaway lists)
        edges: list[tuple] = []
        cap = 200
        for u in nodes | {node}:
            for _, v, key, data in self.G.out_edges(u, keys=True, data=True):
                if v in nodes or v == node:
                    ts = data.get("timestamp")
                    if isinstance(ts, datetime):
                        ts_str = ts.isoformat()
                    elif isinstance(ts, pd.Timestamp):
                        ts_str = ts.isoformat()
                    else:
                        ts_str = str(ts)
                    edges.append((u, v, ts_str, float(data.get("amount", 0.0))))
                    if len(edges) >= cap:
                        return nodes, edges
        return nodes, edges

    def _unique_predecessors(self, node: str) -> int:
        if not self.G.has_node(node):
            return 0
        return len(set(self.G.predecessors(node)))

    def _unique_successors(self, node: str) -> int:
        if not self.G.has_node(node):
            return 0
        return len(set(self.G.successors(node)))

    @staticmethod
    def _fan_score(degree: int, unique: int) -> float:
        """A simple fan score in [0, 1]: 0 when every edge is to a unique
        partner; → 1 when many edges concentrate on a single partner.
        """
        if degree <= 0:
            return 0.0
        if unique <= 0:
            return 0.0
        return 1.0 - (unique / degree)

    def _local_cycle_counts(self, node: str, max_radius: int = 3) -> tuple[int, int]:
        """Count short cycles involving ``node`` within a small radius.

        Heuristic: count directed simple cycles of length ≤ 3 in the induced
        subgraph of ``node`` + 2-hop neighbours. Caps neighbourhood size to
        keep this O(local).
        """
        if not self.G.has_node(node):
            return 0, 0
        nodes = {node}
        nodes.update(self.G.successors(node))
        nodes.update(self.G.predecessors(node))
        if len(nodes) > 30:
            return 0, 0
        for n in list(nodes):
            nodes.update(self.G.successors(n))
            nodes.update(self.G.predecessors(n))
            if len(nodes) > 80:
                break
        sub = self.G.subgraph(nodes)
        c2 = c3 = 0
        try:
            for cycle in nx.simple_cycles(sub, length_bound=3):
                if node in cycle:
                    if len(cycle) == 2:
                        c2 += 1
                    elif len(cycle) == 3:
                        c3 += 1
        except Exception:
            pass
        return c2, c3

    def _community(self, account_id: str) -> dict:
        return self._community_idx.get(account_id, {})

    # ---- R14 helpers ---- #
    def _in_edges_within(self, node: str, current_ts: datetime, window: timedelta):
        if not self.G.has_node(node):
            return []
        cutoff = current_ts - window
        return [
            (u, v, k, d)
            for (u, v, k, d) in self.G.in_edges(node, keys=True, data=True)
            if _to_ts(d.get("timestamp")) >= cutoff
        ]

    # ---- R15 helpers ---- #
    def _in_unique_within(self, node: str, current_ts: datetime, window: timedelta) -> int:
        edges = self._in_edges_within(node, current_ts, window)
        return len({u for (u, _, _, _) in edges})

    def _out_unique_within(self, node: str, current_ts: datetime, window: timedelta) -> int:
        if not self.G.has_node(node):
            return 0
        cutoff = current_ts - window
        return len(
            {
                v
                for (_, v, _, d) in self.G.out_edges(node, keys=True, data=True)
                if _to_ts(d.get("timestamp")) >= cutoff
            }
        )

    def _last_inflow_before(self, node: str, ts: datetime, exclude_edge_key: str | None = None) -> dict | None:
        if not self.G.has_node(node):
            return None
        latest = None
        latest_ts = None
        for (_, _, key, d) in self.G.in_edges(node, keys=True, data=True):
            if exclude_edge_key is not None and key == exclude_edge_key:
                continue
            edge_ts = _to_ts(d.get("timestamp"))
            if edge_ts < ts and (latest_ts is None or edge_ts > latest_ts):
                latest = d
                latest_ts = edge_ts
        return latest


WINDOW_24H = timedelta(hours=24)
WINDOW_7D = timedelta(days=7)
