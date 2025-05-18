# Simulation Skeleton (whereareyou‑DASH)

A minimal Python framework for the whereareyou‑DASH recursive, economy‑anchored reality simulation. Includes core dynamics (ψ‑flux, convergence, stability, reserves, panic), node‑anchor bleed logic, contradiction checking, and GitHub CI.

## 🚀 Features

* **Core Simulation Engine**

  * `Simulation` class with time‑stepped updates for psionic flux (ψ), stability (S), reserves (R), convergence (C), and panic (P).
  * Pluggable `NodeAnchor` bleed modes: linear, exponential, stochastic.
  * Automated contradiction detection & freeze via `FreezeException`.
  * History logging and matplotlib‑powered plotting.

* **Modular Manifest JSON**

  * `upgraded_simulation_overview.json` holds directives, modules, storylines, energy taxonomy, implementation status, and simulation‑module metadata.
  * Easy to ingest via the HF‑4 “Upload Point Interface” in whereareyou‑DASH.

* **GitHub Actions CI**

  * On push/PR: tests across Python 3.10–3.12, installs dependencies, runs the simulation script, and verifies JSON manifest loads.

## 📁 Repository Structure

```text
simulation_skeleton/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions workflow
├── n_skeleton.py              # Core simulation script
├── upgraded_simulation_overview.json
├── upgraded_simulation_with_sim.json
├── requirements.txt           # (optional) libs: matplotlib, etc.
└── README.md
```

## 🛠️ Setup & Installation

1. **Clone** the repo

   ```bash
   git clone https://github.com/your-org/simulation_skeleton.git
   cd simulation_skeleton
   ```

2. **Create & activate** a virtual environment

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .\.venv\Scripts\Activate.ps1 # Windows PowerShell
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

Run the simulation for 10 beats and generate a plot:

```bash
python n_skeleton.py
```

### Logs

* `full_log.txt`: serialized history and any contradictions.
* `simulation_plot.png`: dynamics of C, ψ, S, R, P over time.

### JSON Manifest

Use your own loader to ingest `upgraded_simulation_with_sim.json` into whereareyou‑DASH or any GPT‑plugin layer.

## 🔧 Configuration

* **Initial state**, **weights**, **alphas**, and **anchors** are defined in `n_skeleton.py`’s `main()`—feel free to tweak.
* To adjust story‑trigger thresholds or add new S‑codes, edit your manifest JSON under:

  ```json
  "layered_system_architecture": {
    "layer_2_all_possible_storylines": [ ... ]
  }
  ```

## 📈 GitHub CI

Your GitHub Actions workflow (`.github/workflows/ci.yml`) will:

1. Check out code on push/PR to `main`.
2. Test under Python 3.10–3.12.
3. Install dependencies.
4. Run `python n_skeleton.py`.
5. Verify `upgraded_simulation_with_sim.json` loads without errors.

## 📝 Contributing

1. Fork the repo
2. Create a branch:

   ```bash
   git checkout -b feature/your-idea
   ```
3. Commit your changes:

   ```bash
   git commit -m "Add ..."
   ```
4. Push & open a Pull Request

Please ensure new code is covered by tests or checked under CI.

## 📜 License

This project is released under the MIT License. See `LICENSE` for details.



