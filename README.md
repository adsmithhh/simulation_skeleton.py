# Psionic Simulation Framework

A GitHub repository template for simulating narrative-based psionic metrics. This project provides definitions and code scaffolding to model and simulate the following metrics:

| Symbol | Name                   | Narrative Essence                  | Unit & Measurement            |
| ------ | ---------------------- | ---------------------------------- | ----------------------------- |
| Ψ      | Psionic Potential      | Raw belief materialization power   | p-units (Ψ·s)                 |
| CCI    | Collective Cognition   | Shared mental framework            | cci-index points              |
| DF     | Dream Frequency        | Probability of dream-event merging | df-quanta per cycle           |
| BFR    | Bayesian Feedback Rate | Systemic correction velocity       | bf-rate (updates/sec)         |
| OR     | Ontological Resonance  | Alignment between myth and reality | or-harmonics                  |
| ER     | Entropic Reversion     | Tendency toward disorder           | entropy bits                  |
| SP     | Symbolic Potency       | Cultural-symbol power density      | sp-symbols per narrative unit |
| AS     | Anomaly Stress         | Stress accumulation from anomalies | as-stress units               |
| AW     | Attention Weight       | Focus distribution metric          | aw-points                     |
| CM     | Convergence Momentum   | Rate of narrative convergence      | cm-velocity                   |

## Project Structure

```text
psionic-sim/
├── README.md         # Project overview and metric definitions
├── LICENSE           # Licensing information (MIT)
├── requirements.txt  # Python dependencies
├── setup.py          # Package setup
├── psionic_sim/
│   ├── __init__.py
│   ├── metrics.py    # Class definitions for each metric
│   ├── simulation.py # Core simulation engine
│   └── utils.py      # Utility functions
└── examples/
    └── run_simulation.py # Example usage
```

## Installation

```bash
git clone https://github.com/<username>/psionic-sim.git
cd psionic-sim
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

In `examples/run_simulation.py`, configure your initial parameters and run:

```bash
python examples/run_simulation.py
```

The output will include time-series data for each metric and visualizations.

## Metric Definitions

Each metric is implemented as a Python class in `psionic_sim/metrics.py` with attributes:

* `name`: Full metric name
* `symbol`: Symbolic identifier
* `unit`: Measurement unit
* `value`: Current numeric value

### Example

```python
from psionic_sim.metrics import PsionicPotential

psi = PsionicPotential(initial=0.5)
psi.update(delta_time=1.0)
print(psi.value, psi.unit)  # e.g., 0.52 Ψ·s
```

## Contributing

Contributions welcome! Please open issues or pull requests.

## License

This project is licensed under the MIT License.

2. Process Equations & Feedback Loops

Psionic Flux Update



Drift rate d controls leak.

Ritual input and phi‑coupling supply new belief­energy.

Stability Update



Ensures environmental wear & maintenance trade‑off.

Reserves Update



Consumption E vs. deposit D.

Convergence Update



Combines psionic drive, instability pressure, and story‑burn rate.

Panic Update



Balances rising dissonance, shocks, and ordering forces.

3. Secondary Modules & Interactions

NodeAnchor Bleed: Anchors leak EFS per linear/exponential/stochastic modes.

Energy Taxonomy Interplay: Mapping Ψ‑CCI‑DF‑…‑CM within each beat.

Economic Feedback: M, Q, T, π, Iₚ loops drive ‘value’ genesis.

Goods Validation: Quality multiplier and rejection logic tie back to Ψ, C, S, AS.

Faction Tensions: Directed edges modulating thresholds, causing cross‑impact.

4. Narrative Trigger Logic

Define exact conditions for each S‑code (S‑01 through S‑14), e.g.:

S‑01 (Unity Broadcast): Ψ_t > θ₁ and CCI_t < θ₂ → global cohesion event.

…etc.

5. Integration & Execution Flow

Load Manifest → parse all parameters, thresholds, multipliers.

Initialize State → set Ψ, C, S, R, P, Node EFS, faction stores.

Beat Loop:

Compute updates for metrics (steps 1–5).

Apply NodeAnchor bleeds.

Validate goods, update economy.

Evaluate Story‑trigger conditions.

Run inconsistency checks.

Log & Visualize → history, QA events, dashboard.

Next Steps

Review and flesh out Narrative Trigger Logic for all S‑codes.

Confirm parameter ranges (θ₁, θ₂, etc.) and units.

Map missing interactions: e.g., BFR, OR, ER influences on core updates.

Lock down execution flow ordering and concurrency guarantees.



6. Player Engine Integration

Treat the human player as a distinct subsystem whose discrete actions each tick emit events that drive the core sim variables. This unifies player, NPC, and environmental influences under the same delta-based update loop.

6.1 Define Player Actions & Output Events

Action

Story Effect

Output Event

Investigate Ruins

Seek lore → ups Symbolic Potency (SP)

{type:"investigate", potency:2}

Rally the Guards

Align factions → ups C (Convergence)

{type:"rally", potency:1}

Sabotage the Ward

Undermine order → downs S (Stability)

{type:"sabotage", potency:3}

Appeal to Nobles

Shift will → ups C & Ψ

{type:"appeal", potency:2}

Retreat / Hide

Avoid conflict → downs P (Panic)

{type:"hide", potency:2}

6.2 Simulation Tick with Player Events

Incorporate external_events from the Player Engine before running the feedback loop:

def simulation_tick(state, external_events):
    # 1. NPC/environment updates
    state = update_from_npcs_and_environment(state)
    # 2. Player actions
    for evt in external_events:
        state = apply_player_event(state, evt)
    # 3. Core feedback loops
    state = feedback_loop_step(state)
    return state

6.3 apply_player_event

Translate player events into variable deltas:

def apply_player_event(state, evt):
    if evt['type']=='investigate':
        state['SP'] += evt['potency'] * state.get('knowledge_factor',1)
    elif evt['type']=='rally':
        state['C']  += evt['potency'] * state.get('faction_cohesion',1)
    elif evt['type']=='sabotage':
        state['S']  -= evt['potency'] * state.get('ward_strength',1)
    elif evt['type']=='appeal':
        state['C']   += evt['potency']*0.5
        state['psi'] += evt['potency']*0.5
    elif evt['type']=='hide':
        state['P']  -= evt['potency']*0.7
    return state

Tune the multipliers so that player influence is visible but balanced against autonomous forces.

6.4 Balancing & Feedback

Action Budget: Limit total potency per tick or introduce resource costs (e.g. Stamina) to prevent spamming.

Contextual Availability: Only offer actions when state thresholds or story triggers allow.

UI Projections: Display projected deltas next to each choice: “Investigate → +2 SP, +0.5 AS.”

Cooldowns & Costs: Assign cooldown periods or currency costs to high-impact actions.

6.5 Example Turn Flow

Render State: Show current C, Ψ, S, R, P, SP, AS, etc.

List Actions: Contextual menu of player actions with projected impacts.

Player Chooses: Emit selected event.

Run Tick: simulation_tick(state, [event]).

Narrative Feedback: Generate a text beat: “Your sabotage cracked the ward—stability falls by 3.”



7. Meta-Architectural Rigor & Directive Enforcement

Our simulation’s architecture mirrors a hyper-structured, pattern-driven cognition—ensuring that every input, module, and output is rigorously filtered, tagged, and self-validated:

Directive Adherence Core: The first-pass gate through which all user inputs, player events, and external data must pass. It enforces the primary mission statements and blocks any unaligned signals before further processing.

Specialized Engine Cascade: Subsequent layers (Φ‑Density, CCI, DASH, PDS‑1, Manifestation, etc.) each operate on pre‑validated inputs, preventing unsanctioned “jumps” or emergent drift.

Pattern-Integrity Enforcer: Continuously scans lexical and semantic coherence across narrative and code modules, flagging any deviations from defined schemas.

Recursive Conflict Resolver: On detecting ambivalent or contradictory states, it interjects with FreezeExceptions or corrective adjustments—maintaining internal logical consistency.

Meta-Awareness Monitor: Aggregates health metrics from all sub-engines and surfaces meta‑insights (e.g., rising paradox rates, over-indexed anomaly stress), guiding high‑level interventions.

This layered, self‑monitoring structure guarantees that simulation outputs are not random noise but the result of a meticulously calibrated cascade of checks and balances.

End of Theoretical Mechanisms Specification

