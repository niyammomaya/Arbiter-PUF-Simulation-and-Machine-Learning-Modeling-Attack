import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from puf_simulator import ArbiterPUF

class PUFModelingAttack:
    def __init__(self, puf):
        self.puf = puf
        self.model = LogisticRegression(max_iter=1000)
        self.is_trained = False

    def _build_feature_matrix(self, challenges):
        return np.array([self.puf._challenge_to_phi(c) for c in challenges])

    def train(self, challenges, responses):
        X = self._build_feature_matrix(challenges)
        self.model.fit(X, responses)
        self.is_trained = True

    def predict(self, challenges):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting.")
        X = self._build_feature_matrix(challenges)
        return self.model.predict(X)

    def evaluate(self, challenges, true_responses):
        predictions = self.predict(challenges)
        return accuracy_score(true_responses, predictions)


if __name__ == "__main__":
    print("=== ML Modeling Attack - Phase 2 ===\n")

    puf = ArbiterPUF(n_stages=64, seed=42)

    print("Collecting CRPs from target PUF...")
    challenges, responses = puf.generate_crps(10000)

    X_train, X_test, y_train, y_test = train_test_split(
        challenges, responses, test_size=0.2, random_state=42
    )
    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")

    print("\nTraining logistic regression model...")
    attack = PUFModelingAttack(puf)
    attack.train(X_train, y_train)

    accuracy = attack.evaluate(X_test, y_test)
    print(f"\nAttack accuracy on unseen challenges: {accuracy:.2%}")

    if accuracy >= 0.95:
        print(">> PUF successfully cloned! (>=95% accuracy)")
    elif accuracy >= 0.85:
        print(">> Strong attack — PUF largely compromised (>=85% accuracy)")
    else:
        print(">> Attack partially successful — more CRPs may improve accuracy")

    print("\nSample predictions vs true responses:")
    print(f"{'Challenge (first 8 bits)':<30} {'True':>6} {'Predicted':>10} {'Match':>6}")
    print("-" * 56)
    for i in range(8):
        true = y_test[i]
        pred = attack.predict(X_test[i:i+1])[0]
        match = "✓" if true == pred else "✗"
        print(f"{str(X_test[i][:8]):<30} {true:>6} {pred:>10} {match:>6}")
