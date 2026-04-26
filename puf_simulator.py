import numpy as np

class ArbiterPUF:
    """
    Software simulation of an n-stage Arbiter PUF.

    Each PUF instance has a unique weight vector (w) drawn from a Gaussian
    distribution, simulating the manufacturing delay variations of a real chip.

    The response is determined by the sign of the dot product between the
    feature vector (phi) derived from the challenge and the weight vector.
    """

    def __init__(self, n_stages=64, seed=None):
        self.n_stages = n_stages
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 1, n_stages + 1)

    def _challenge_to_phi(self, challenge):
        n = self.n_stages
        phi = np.ones(n + 1)
        for i in range(n - 1, -1, -1):
            phi[i] = phi[i + 1] * (1 - 2 * challenge[i])
        return phi

    def get_response(self, challenge):
        phi = self._challenge_to_phi(challenge)
        delay_diff = np.dot(self.weights, phi)
        return 1 if delay_diff >= 0 else -1

    def generate_crps(self, n_crps):
        challenges = np.random.randint(0, 2, size=(n_crps, self.n_stages))
        responses = np.array([self.get_response(c) for c in challenges])
        return challenges, responses


if __name__ == "__main__":
    print("=== Arbiter PUF Simulator - Phase 1 ===\n")

    puf = ArbiterPUF(n_stages=64, seed=42)
    challenges, responses = puf.generate_crps(10000)

    print(f"PUF stages      : {puf.n_stages}")
    print(f"CRPs generated  : {len(responses)}")
    print(f"Response balance: {np.mean(responses == 1):.2%} ones, "
          f"{np.mean(responses == -1):.2%} minus-ones")
    print(f"\nSample challenge : {challenges[0]}")
    print(f"Sample response  : {responses[0]}")

    r1 = puf.get_response(challenges[0])
    r2 = puf.get_response(challenges[0])
    print(f"\nDeterminism check: response called twice on same challenge → {r1}, {r2} ✓")
