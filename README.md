# Arbiter PUF Simulation and ML Modeling Attack

A software simulation of an Arbiter Physical Unclonable Function (PUF) and a machine learning-based modeling attack against it. Built as a final project for a hardware security course at Rutgers University.

The core idea: Arbiter PUFs are supposed to be unclonable because they rely on physical manufacturing variations unique to each chip. This project shows that a standard logistic regression classifier trained on a few hundred observed challenge-response pairs can predict the PUF's responses with ~99% accuracy — effectively cloning it in software.

---

## Files

| File | Description |
|------|-------------|
| `puf_simulator.py` | Implements the `ArbiterPUF` class. Simulates an n-stage Arbiter PUF and generates challenge-response pairs (CRPs). |
| `puf_attack.py` | Implements the `PUFModelingAttack` class. Trains a logistic regression model on observed CRPs and evaluates cloning accuracy on unseen challenges. |
| `puf_evaluation.py` | Runs three experiments and generates plots: accuracy vs. training size, accuracy vs. stage count, and logistic regression vs. SVM comparison. |
| `puf_demo.py` | Interactive terminal demo. Walks through the full attack step by step with live output — good for demonstrations. |

---

## Requirements

Python 3 and the following libraries:

```bash
pip install numpy scikit-learn matplotlib
```

No GPU or special hardware required. Tested on Ubuntu 24.04.

---

## How to Run

### Run the full evaluation (generates all 3 plots)

```bash
python3 puf_evaluation.py
```

This will print results to the terminal and save three figures to the working directory:

- `plot1_accuracy_vs_crps.png` — attack accuracy vs. number of training CRPs
- `plot2_accuracy_vs_stages.png` — attack accuracy vs. PUF stage count
- `plot3_lr_vs_svm.png` — logistic regression vs. SVM comparison

Takes under 60 seconds on a standard laptop.

### Run the simulator on its own

```bash
python3 puf_simulator.py
```

Creates a 64-stage PUF, generates 10,000 CRPs, and runs a quick sanity check.

### Run the attack on its own

```bash
python3 puf_attack.py
```

Generates CRPs from a 64-stage PUF, trains logistic regression on 80% of them, and reports accuracy on the held-out 20%.

### Run the live demo

```bash
python3 puf_demo.py
```

Interactive step-by-step walkthrough of the full attack. Press Enter to advance through each stage. Useful for presentations or screen recordings.

---

## Results

| Experiment | Result |
|---|---|
| Attack accuracy (8,000 training CRPs, 64-stage PUF) | **99.35%** |
| CRPs needed to reach 95% accuracy | **~500** |
| Accuracy drop from 16 → 128 stages | **< 1%** |
| LR vs. SVM difference at 8,000 CRPs | **negligible** |

---

## Reproducing the Paper Results

The PUF random seed is fixed at `42` throughout all experiments, so results are fully reproducible. Just run `puf_evaluation.py` and the output should match the numbers in the report.

---

## Project Structure

```
puf_project/
├── puf_simulator.py     # PUF simulation
├── puf_attack.py        # ML modeling attack
├── puf_evaluation.py    # Experiments and plots
├── puf_demo.py          # Interactive demo
└── README.md
```

---

## How It Works (Short Version)

An Arbiter PUF races two electrical signals through n stages. At each stage, a challenge bit decides whether the paths cross or pass straight. The arbiter at the end records which path wins — that's the response.

The vulnerability: the response can be written as `sign(w · Φ(c))`, where `Φ` is a public transformation of the challenge bits and `w` is the chip's secret weight vector. This is exactly what logistic regression learns — a linear decision boundary. So an attacker who collects enough challenge-response pairs can train a model that approximates `w` and predicts future responses without ever touching the hardware.

---

*Hardware Security — Rutgers University ECE*
