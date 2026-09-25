"""
Aegis — Agentic Orchestrator
-----------------------------
Three cooperating agents coordinated through a shared, typed state graph
(LangGraph). Each agent has a single responsibility and hands control to
the next based on graph edges — no single monolithic prompt is doing
everything, which keeps decisions auditable and independently testable.

Agents:
  1. TriageAgent   — ranks affected zones by severity + population risk
  2. ResourceAgent — allocates ambulances / rescue teams / shelters
  3. CommsAgent    — drafts public alerts and responder instructions,
                      grounded via RAG against real emergency protocols
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict, List, Dict

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

from rag_client import RAGClient  # calls the RAG service (see rag-service/)


class ZoneAssessment(TypedDict):
    zone_id: str
    severity: str
    confidence: float
    population_estimate: int
    flood_extent_pct: float


class AegisState(TypedDict):
    zones: List[ZoneAssessment]
    priority_order: List[str]
    allocations: Dict[str, List[str]]      # zone_id -> resource ids
    available_resources: Dict[str, int]     # resource_type -> count
    alerts: List[str]
    log: Annotated[List[str], operator.add]


SEVERITY_WEIGHT = {"destroyed": 4, "major-damage": 3, "minor-damage": 2, "no-damage": 0}


def triage_agent(state: AegisState) -> AegisState:
    """Rank zones using severity, model confidence, and estimated population at risk."""
    def score(z: ZoneAssessment) -> float:
        base = SEVERITY_WEIGHT.get(z["severity"], 0) * z["confidence"]
        pop_factor = min(z["population_estimate"] / 1000, 5)
        return base + pop_factor + z["flood_extent_pct"] / 100

    ranked = sorted(state["zones"], key=score, reverse=True)
    state["priority_order"] = [z["zone_id"] for z in ranked]
    state["log"] = [f"[Triage] Ranked {len(ranked)} zones; top priority: {state['priority_order'][:3]}"]
    return state


def resource_agent(state: AegisState) -> AegisState:
    """Greedy allocation of limited resources to highest-priority zones first."""
    remaining = dict(state["available_resources"])
    allocations: Dict[str, List[str]] = {}

    for zone_id in state["priority_order"]:
        zone = next(z for z in state["zones"] if z["zone_id"] == zone_id)
        needed = []
        if zone["severity"] in ("destroyed", "major-damage") and remaining.get("rescue_teams", 0) > 0:
            remaining["rescue_teams"] -= 1
            needed.append("rescue_team")
        if zone["flood_extent_pct"] > 30 and remaining.get("boats", 0) > 0:
            remaining["boats"] -= 1
            needed.append("boat")
        if zone["population_estimate"] > 500 and remaining.get("ambulances", 0) > 0:
            remaining["ambulances"] -= 1
            needed.append("ambulance")
        allocations[zone_id] = needed

    state["allocations"] = allocations
    state["available_resources"] = remaining
    state["log"] = [f"[Resource] Allocated resources across {len(allocations)} zones; remaining: {remaining}"]
    return state


def comms_agent(state: AegisState) -> AegisState:
    """Draft grounded alerts using RAG-retrieved protocol snippets, not free-form generation."""
    rag = RAGClient()
    alerts = []
    for zone_id in state["priority_order"][:5]:
        zone = next(z for z in state["zones"] if z["zone_id"] == zone_id)
        protocol_snippets = rag.retrieve(
            query=f"emergency response protocol for {zone['severity']} damage with flooding"
        )
        alloc = state["allocations"].get(zone_id, [])
        alert = (
            f"ZONE {zone_id}: {zone['severity'].upper()} — "
            f"{', '.join(alloc) if alloc else 'monitoring only'}. "
            f"Guidance basis: {protocol_snippets[0]['source'] if protocol_snippets else 'general SOP'}."
        )
        alerts.append(alert)

    state["alerts"] = alerts
    state["log"] = [f"[Comms] Drafted {len(alerts)} grounded alerts"]
    return state


def build_graph() -> StateGraph:
    graph = StateGraph(AegisState)
    graph.add_node("triage", triage_agent)
    graph.add_node("resources", resource_agent)
    graph.add_node("comms", comms_agent)

    graph.set_entry_point("triage")
    graph.add_edge("triage", "resources")
    graph.add_edge("resources", "comms")
    graph.add_edge("comms", END)
    return graph.compile()


aegis_graph = build_graph()


def run_response_cycle(zones: List[ZoneAssessment], available_resources: Dict[str, int]) -> AegisState:
    initial_state: AegisState = {
        "zones": zones,
        "priority_order": [],
        "allocations": {},
        "available_resources": available_resources,
        "alerts": [],
        "log": [],
    }
    return aegis_graph.invoke(initial_state)
