import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from puf_simulator import ArbiterPUF
from puf_attack import PUFModelingAttack


# ── Plot 1: Accuracy vs Number of Training CRPs ────────────────────────────
def plot_accuracy_vs_crps():
    print("Running Plot 1: Accuracy vs Training CRPs...")

    puf = ArbiterPUF(n_stages=64, seed=42)
    all_challenges, all_responses = puf.generate_crps(20000)

    X_test = all_challenges[-2000:]
    y_test = all_responses[-2000:]
    X_pool = all_challenges[:-2000]
    y_pool = all_responses[:-2000]

    training_sizes = [100, 500, 1000, 2000, 4000, 6000, 8000, 10000, 15000, 18000]
    accuracies = []

    for n in training_sizes:
        attack = PUFModelingAttack(puf)
        attack.train(X_pool[:n], y_pool[:n])
        acc = attack.evaluate(X_test, y_test)
        accuracies.append(acc * 100)
        print(f"  Training size: {n:>6} → Accuracy: {acc:.2%}")

    plt.figure(figsize=(9, 5))
    plt.plot(training_sizes, accuracies, marker='o', color='steelblue',
             linewidth=2, markersize=6)
    plt.axhline(y=95, color='red', linestyle='--', linewidth=1.2, label='95% threshold')
    plt.fill_between(training_sizes, accuracies, alpha=0.1, color='steelblue')
    plt.xlabel("Number of Training CRPs", fontsize=12)
    plt.ylabel("Attack Accuracy (%)", fontsize=12)
    plt.title("ML Modeling Attack Accuracy vs Number of Training CRPs\n(64-stage Arbiter PUF)", fontsize=13)
    plt.legend(fontsize=11)
    plt.xticks(training_sizes, rotation=45)
    plt.ylim(50, 101)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot1_accuracy_vs_crps.png", dpi=150)
    plt.close()
    print("  Saved → plot1_accuracy_vs_crps.png\n")


# ── Plot 2: Accuracy vs PUF Stage Size ─────────────────────────────────────
def plot_accuracy_vs_stages():
    print("Running Plot 2: Accuracy vs PUF Stage Size...")

    stage_sizes = [16, 32, 48, 64, 96, 128]
    training_size = 8000
    test_size = 2000
    accuracies = []

    for n_stages in stage_sizes:
        puf = ArbiterPUF(n_stages=n_stages, seed=42)
        challenges, responses = puf.generate_crps(training_size + test_size)

        X_train = challenges[:training_size]
        y_train = responses[:training_size]
        X_test  = challenges[training_size:]
        y_test  = responses[training_size:]

        attack = PUFModelingAttack(puf)
        attack.train(X_train, y_train)
        acc = attack.evaluate(X_test, y_test)
        accuracies.append(acc * 100)
        print(f"  Stages: {n_stages:>4} → Accuracy: {acc:.2%}")

    plt.figure(figsize=(9, 5))
    plt.plot(stage_sizes, accuracies, marker='s', color='darkorange',
             linewidth=2, markersize=6)
    plt.axhline(y=95, color='red', linestyle='--', linewidth=1.2, label='95% threshold')
    plt.fill_between(stage_sizes, accuracies, alpha=0.1, color='darkorange')
    plt.xlabel("Number of PUF Stages", fontsize=12)
    plt.ylabel("Attack Accuracy (%)", fontsize=12)
    plt.title("ML Modeling Attack Accuracy vs PUF Stage Size\n(8,000 Training CRPs)", fontsize=13)
    plt.legend(fontsize=11)
    plt.xticks(stage_sizes)
    plt.ylim(50, 101)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot2_accuracy_vs_stages.png", dpi=150)
    plt.close()
    print("  Saved → plot2_accuracy_vs_stages.png\n")


# ── Plot 3: Logistic Regression vs SVM ─────────────────────────────────────
def plot_lr_vs_svm():
    print("Running Plot 3: Logistic Regression vs SVM...")

    puf = ArbiterPUF(n_stages=64, seed=42)
    all_challenges, all_responses = puf.generate_crps(20000)

    X_test = all_challenges[-2000:]
    y_test = all_responses[-2000:]
    X_pool = all_challenges[:-2000]
    y_pool = all_responses[:-2000]

    training_sizes = [100, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    lr_accuracies  = []
    svm_accuracies = []

    for n in training_sizes:
        X_train = X_pool[:n]
        y_train = y_pool[:n]

        X_train_phi = np.array([puf._challenge_to_phi(c) for c in X_train])
        X_test_phi  = np.array([puf._challenge_to_phi(c) for c in X_test])

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_phi, y_train)
        lr_acc = accuracy_score(y_test, lr.predict(X_test_phi)) * 100
        lr_accuracies.append(lr_acc)

        # SVM with linear kernel + feature scaling to avoid convergence issues
        svm = make_pipeline(StandardScaler(), SVC(kernel='linear', max_iter=10000))
        svm.fit(X_train_phi, y_train)
        svm_acc = accuracy_score(y_test, svm.predict(X_test_phi)) * 100
        svm_accuracies.append(svm_acc)

        print(f"  Training size: {n:>6} → LR: {lr_acc:.2f}%  SVM: {svm_acc:.2f}%")

    plt.figure(figsize=(9, 5))
    plt.plot(training_sizes, lr_accuracies,  marker='o', color='steelblue',
             linewidth=2, markersize=6, label='Logistic Regression')
    plt.plot(training_sizes, svm_accuracies, marker='s', color='seagreen',
             linewidth=2, markersize=6, label='SVM (linear kernel)')
    plt.axhline(y=95, color='red', linestyle='--', linewidth=1.2, label='95% threshold')
    plt.xlabel("Number of Training CRPs", fontsize=12)
    plt.ylabel("Attack Accuracy (%)", fontsize=12)
    plt.title("Logistic Regression vs SVM Attack Accuracy\n(64-stage Arbiter PUF)", fontsize=13)
    plt.legend(fontsize=11)
    plt.xticks(training_sizes, rotation=45)
    plt.ylim(50, 101)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot3_lr_vs_svm.png", dpi=150)
    plt.close()
    print("  Saved → plot3_lr_vs_svm.png\n")


# ── Run all plots ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Evaluation & Visualization - Phase 3 ===\n")
    plot_accuracy_vs_crps()
    plot_accuracy_vs_stages()
    plot_lr_vs_svm()
    print("All plots generated successfully!")
