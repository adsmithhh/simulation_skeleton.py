import random
import math
import matplotlib.pyplot as plt
import datetime
import os
import json

class FreezeException(Exception):
    pass

class NodeAnchor:
    def __init__(self, mode, params):
        self.mode = mode
        self.params = params

    def bleed(self, t, efs):
        if self.mode == 'linear':
            r = self.params.get('r', 0.0)
            return max(0.0, efs - r)
        elif self.mode == 'exponential':
            h = self.params.get('h', 1.0)
            return efs * (2 ** (-1.0 / h))
        elif self.mode == 'stochastic':
            drops = self.params.get('drops', [0.0])
            probs = self.params.get('probs', [1.0])
            r = random.choices(drops, probs)[0]
            return max(0.0, efs - r)
        else:
            return efs

class Simulation:
    def __init__(self, initial_state, weights, alphas, anchors, psi_max, faction):
        self.state = initial_state.copy()
        self.weights = weights
        self.alphas = alphas
        self.anchors = anchors
        self.psi_max = psi_max
        self.beat = 0
        self.last_inputs = {}
        self.history = []
        self.contradictions = []
        self.faction = faction  # <-- Add this line

    def update_flux(self, d, I_ritual, R_phi):
        psi = self.state['psi']
        return psi * (1 - d) + I_ritual * R_phi

    def update_stability(self, S, drift, repair):
        return S - drift + repair

    def update_reserves(self, R, E, D, R_max):
        return max(0, R - E + D)

    def update_convergence(self, C, psi, S, E, R_max):
        w_psi = self.weights['w_psi']
        w_OM = self.weights['w_OM']
        w_SB = self.weights['w_SB']
        term1 = w_psi * (1 - math.exp(-psi / self.psi_max))
        term2 = w_OM * (1 - S)
        term3 = w_SB * (E / R_max)
        return C + term1 + term2 + term3

    def update_panic(self, P, C_old, C_new, S_shock, O_action):
        alpha_C = self.alphas['alpha_C']
        alpha_shock = self.alphas['alpha_shock']
        alpha_order = self.alphas['alpha_order']
        delta = alpha_C * (C_new - C_old) + alpha_shock * S_shock - alpha_order * O_action
        return min(10, max(0, P + delta))

    def contradiction_check(self):
        contradictions = []
        if self.state['S'] < 0:
            contradictions.append("Stability dropped below zero.")
        if self.state['R'] < self.last_inputs.get('E', 0) and self.last_inputs.get('D', 0) == 0:
            contradictions.append("Reserve depleted without gain.")
        if contradictions:
            contradiction_record = {
                'beat': self.beat,
                'state': self.state.copy(),
                'inputs': self.last_inputs.copy(),
                'reasons': contradictions
            }
            self.contradictions.append(contradiction_record)
            raise FreezeException("; ".join(contradictions))

    def step(self, inputs):
        self.beat += 1
        self.last_inputs = inputs
        C_old = self.state['C']

        self.state['psi'] = self.update_flux(inputs['d'], inputs['I_ritual'], inputs['R_phi'])
        self.state['S'] = self.update_stability(self.state['S'], inputs['drift'], inputs['repair'])
        self.state['R'] = self.update_reserves(self.state['R'], inputs['E'], inputs['D'], inputs['R_max'])
        C_new = self.update_convergence(self.state['C'], self.state['psi'], self.state['S'], inputs['E'], inputs['R_max'])
        self.state['C'] = C_new
        self.state['P'] = self.update_panic(self.state['P'], C_old, C_new, inputs['S_shock'], inputs['O_action'])

        for node, anchor in self.anchors.items():
            self.state[node]['EFS'] = anchor.bleed(self.beat, self.state[node]['EFS'])

        self.history.append({
            'beat': self.beat,
            'state': self.state.copy(),
            'inputs': inputs.copy()
        })

        self.contradiction_check()
        return self.state

def plot_simulation(history):
    beats = [entry['beat'] for entry in history]
    C_vals = [entry['state']['C'] for entry in history]
    psi_vals = [entry['state']['psi'] for entry in history]
    S_vals = [entry['state']['S'] for entry in history]
    R_vals = [entry['state']['R'] for entry in history]
    P_vals = [entry['state']['P'] for entry in history]

    plt.figure(figsize=(10, 6))
    plt.plot(beats, C_vals, label='C (Convergence)')
    plt.plot(beats, psi_vals, label='ψ (Flux)')
    plt.plot(beats, S_vals, label='S (Stability)')
    plt.plot(beats, R_vals, label='R (Reserves)')
    plt.plot(beats, P_vals, label='P (Panic)')

    plt.xlabel('Beat')
    plt.ylabel('Value')
    plt.title('Simulation Dynamics Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save with timestamp in sim_plots folder
    folder = "sim_plots"
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"simulation_plot_{timestamp}.png")
    plt.savefig(filename)
    plt.show()
    return filename

def plot_faction_comparison(factions, initial_state, weights, alphas, anchors, psi_max, inputs, beats=10):
    plt.figure(figsize=(10, 6))
    for faction_id, faction in factions.items():
        sim = Simulation(initial_state, weights, alphas, anchors, psi_max, faction)
        try:
            for _ in range(beats):
                sim.step(inputs)
        except FreezeException:
            pass
        C_vals = [entry['state']['C'] for entry in sim.history]
        plt.plot(range(1, len(C_vals)+1), C_vals, label=f"{faction_id}: {faction['name']}")
    plt.xlabel('Beat')
    plt.ylabel('C (Convergence)')
    plt.title('Convergence Comparison Across Factions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    folder = "sim_plots"
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"faction_comparison_{timestamp}.png")
    plt.savefig(filename)
    plt.show()
    return filename  # <-- Add this line

def load_factions(filename="faction.json"):
    with open(filename) as f:
        data = json.load(f)
    return {f['id']: f for f in data['factions']}

def get_faction(faction_id, factions):
    return factions.get(faction_id, None)

def validate_goods(record, factions):
    faction = get_faction(record['producer'], factions)
    if not faction:
        record['reasons'].append('Unknown faction')
        return False

    # Apply multipliers
    record['psionic_signature'] *= faction['psi_multiplier']
    record['convergence_score'] += faction['cci_adjustment']

    # Use faction-specific thresholds
    thresholds = faction['validation_thresholds']
    if record['psionic_signature'] < thresholds['Psi_min']:
        record['reasons'].append('Ψ below faction-min')
    if record['convergence_score'] < thresholds['C_min']:
        record['reasons'].append('C below faction-min')
    if record['stability'] < thresholds['S_min']:
        record['reasons'].append('S below faction-min')
    if record['anomaly_score'] > thresholds['AS_max']:
        record['reasons'].append('AS above faction-max')

    return len(record['reasons']) == 0

def update_currency_supply(M, psi, faction):
    beta_psi = 1.0  # Example coefficient
    return M + beta_psi * (psi * faction['psi_multiplier']) * faction['investment_pref']

def main():
    factions = load_factions("faction.json")  # No argument needed, uses the full path by default
    faction = factions['faction-01']

    initial_state = {
        'C': 0.1, 'psi': 11, 'S': 0.1, 'R': 200, 'P': 1,
        'Node-12': {'EFS': 29},
        'Node-51': {'EFS': 99},
        'Node-23': {'EFS': 18}
    }
    weights = {'w_psi': 0.9, 'w_OM': 0.6, 'w_SB': 0.01}
    alphas = {'alpha_C': 1.0, 'alpha_shock': 1.5, 'alpha_order': 0.5}
    anchors = {
        'Node-12': NodeAnchor('linear', {'r': 0.053}),
        'Node-51': NodeAnchor('exponential', {'h': 4.8}),
        'Node-23': NodeAnchor('stochastic', {'drops': [0.03, 0.05, 0.07], 'probs': [0.5, 0.3, 0.2]})
    }
    sim = Simulation(initial_state, weights, alphas, anchors, psi_max=100, faction=faction)

    inputs = {
        'd': 0.29, 'I_ritual': 1, 'R_phi': 0.3,
        'drift': 0.11, 'repair': 0.95,
        'E': 1, 'D': 1, 'R_max': 250,
        'S_shock': 0.9, 'O_action': 0.1
    }

    try:
        for _ in range(10):
            sim.step(inputs)
    except FreezeException as e:
        print(f"❄️ Simulation frozen: {e}")

    with open("full_log.txt", "w") as f:
        for entry in sim.history:
            f.write(f"{entry}\n")
        if sim.contradictions:
            f.write("\n--- Contradictions ---\n")
            for entry in sim.contradictions:
                f.write(f"{entry}\n")

    plot_file = plot_simulation(sim.history)
    print(f"Plot saved as {plot_file}")

    # Only compare Governance Resonance and Iron Accord
    factions_to_compare = {
      k: v for k, v in factions.items() if k in ['faction-01', 'faction-02', 'faction-03', 'faction-04']

    }
    comparison_file = plot_faction_comparison(
        factions_to_compare,
        initial_state,
        weights,
        alphas,
        anchors,
        psi_max=100,
        inputs=inputs,
        beats=10
    )
    print(f"Faction comparison plot saved as {comparison_file}")

    # Extract and save only the beats information to a new file
    input_file = "simulation_output.txt"
    output_file = "simulation_output_clean.txt"

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if "Beat" in line:
                outfile.write(line)

    print(f"Cleaned output saved to {output_file}")

if __name__ == '__main__':
    main()
