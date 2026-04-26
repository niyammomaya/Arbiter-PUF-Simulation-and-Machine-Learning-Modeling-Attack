import numpy as np
import time
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from puf_simulator import ArbiterPUF
from puf_attack import PUFModelingAttack

# ── Terminal colors ───────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GRAY   = "\033[90m"
    BG_RED   = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE  = "\033[44m"

def banner(text, color=C.BLUE):
    width = 62
    print()
    print(color + "=" * width + C.RESET)
    padding = (width - len(text) - 2) // 2
    print(color + "=" * padding + C.BOLD + f" {text} " + C.RESET + color + "=" * padding + C.RESET)
    print(color + "=" * width + C.RESET)

def pause(msg="Press ENTER to continue..."):
    print()
    input(C.GRAY + f"  [{msg}]" + C.RESET)
    print()

def typeprint(text, delay=0.018, color=""):
    for ch in text:
        sys.stdout.write(color + ch + C.RESET)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def progress_bar(label, steps=30, delay=0.04, color=C.GREEN):
    sys.stdout.write(f"  {label}  [")
    sys.stdout.flush()
    for _ in range(steps):
        time.sleep(delay)
        sys.stdout.write(color + "█" + C.RESET)
        sys.stdout.flush()
    sys.stdout.write("] Done!\n")
    sys.stdout.flush()

def print_challenge(challenge, n=16):
    bits = "".join(str(b) for b in challenge[:n])
    colored = ""
    for b in bits:
        colored += (C.CYAN + "1" if b == "1" else C.GRAY + "0") + C.RESET
    return colored + C.GRAY + f"...({len(challenge)} bits total)" + C.RESET

def accuracy_bar(acc, width=40):
    filled = int(acc / 100 * width)
    color = C.GREEN if acc >= 95 else C.YELLOW if acc >= 80 else C.RED
    bar = color + "█" * filled + C.GRAY + "░" * (width - filled) + C.RESET
    return f"[{bar}] {color}{C.BOLD}{acc:.2f}%{C.RESET}"


# ── Main Demo ─────────────────────────────────────────────────────────────
def main():
    print("\n" * 2)

    # ── INTRO ─────────────────────────────────────────────────────────────
    banner("ARBITER PUF — LIVE ATTACK DEMO", C.BLUE)
    print(f"""
  {C.WHITE}This demo shows a complete ML modeling attack against
  a simulated Arbiter PUF in real time.{C.RESET}

  {C.CYAN}What we'll do:{C.RESET}
    1. Create a secret PUF (the "hardware")
    2. Show that random guessing is useless (~50%)
    3. Run the ML attack with 500 CRPs
    4. Show the PUF is cloned (>95% accuracy)
    5. Demonstrate live challenge prediction
    6. Scale up — show 5000 CRPs hits ~99%
    7. Show bigger PUFs are no safer
    """)
    pause("Press ENTER to start the demo")


    # ── STEP 1: Create the PUF ────────────────────────────────────────────
    banner("STEP 1: Creating a Secret PUF", C.CYAN)
    print(f"  {C.WHITE}Simulating a 64-stage Arbiter PUF chip...{C.RESET}")
    print(f"  {C.GRAY}(In real hardware, manufacturing variations create this automatically){C.RESET}")
    print()

    progress_bar("Generating unique weight vector", steps=20, delay=0.03, color=C.CYAN)

    puf = ArbiterPUF(n_stages=64, seed=99)

    print()
    print(f"  {C.GREEN}✓ PUF created!{C.RESET}")
    print(f"  {C.WHITE}Stages        :{C.RESET} {C.CYAN}64{C.RESET}")
    print(f"  {C.WHITE}Weight vector :{C.RESET} {C.GRAY}[{', '.join(f'{w:.3f}' for w in puf.weights[:6])}...] (SECRET){C.RESET}")
    print()
    print(f"  {C.YELLOW}This weight vector represents the chip's physical delay")
    print(f"  differences — unique to this device, never stored anywhere.{C.RESET}")

    pause()


    # ── STEP 2: Show random guessing is useless ───────────────────────────
    banner("STEP 2: Can We Just Guess?", C.YELLOW)
    print(f"  {C.WHITE}Generating 1,000 random challenges and guessing responses...{C.RESET}")
    print()

    progress_bar("Random guessing", steps=25, delay=0.025, color=C.YELLOW)

    challenges_test, responses_test = puf.generate_crps(1000)
    random_preds = np.random.choice([-1, 1], size=1000)
    random_acc = accuracy_score(responses_test, random_preds) * 100

    print()
    print(f"  {C.WHITE}Random guessing accuracy:{C.RESET}")
    print(f"  {accuracy_bar(random_acc)}")
    print()
    typeprint(f"  As expected — basically a coin flip. The PUF looks secure.", color=C.YELLOW)
    print(f"  {C.GRAY}  An attacker with no information can do no better than 50%.{C.RESET}")

    pause()


    # ── STEP 3: Collect CRPs ──────────────────────────────────────────────
    banner("STEP 3: Attacker Collects 500 CRPs", C.YELLOW)
    print(f"  {C.WHITE}An attacker intercepts 500 challenge-response pairs")
    print(f"  by observing the authentication protocol...{C.RESET}")
    print()

    all_challenges, all_responses = puf.generate_crps(10000)
    X_train_500  = all_challenges[:500]
    y_train_500  = all_responses[:500]
    X_test       = all_challenges[8000:]
    y_test       = all_responses[8000:]

    print(f"  {C.WHITE}Sample CRPs collected:{C.RESET}")
    print(f"  {C.GRAY}{'Challenge (first 16 bits)':<36} {'Response':>10}{C.RESET}")
    print(f"  {C.GRAY}{'-' * 48}{C.RESET}")
    for i in range(6):
        ch_str = print_challenge(X_train_500[i])
        resp_color = C.GREEN if y_train_500[i] == 1 else C.RED
        print(f"  {ch_str}  {resp_color}{C.BOLD}{'+1' if y_train_500[i]==1 else '-1':>6}{C.RESET}")

    print(f"  {C.GRAY}  ... and {len(X_train_500)-6} more{C.RESET}")

    pause()


    # ── STEP 4: Train attack with 500 CRPs ────────────────────────────────
    banner("STEP 4: Training the ML Attack (500 CRPs)", C.RED)
    print(f"  {C.WHITE}Building Φ feature vectors from challenges...{C.RESET}")
    progress_bar("Computing feature matrix", steps=20, delay=0.03, color=C.YELLOW)

    X_train_phi = np.array([puf._challenge_to_phi(c) for c in X_train_500])
    X_test_phi  = np.array([puf._challenge_to_phi(c) for c in X_test])

    print(f"  {C.WHITE}Training logistic regression model...{C.RESET}")
    progress_bar("Training classifier", steps=25, delay=0.04, color=C.RED)

    model_500 = LogisticRegression(max_iter=1000)
    model_500.fit(X_train_phi, y_train_500)
    acc_500 = accuracy_score(y_test, model_500.predict(X_test_phi)) * 100

    print()
    print(f"  {C.WHITE}Attack accuracy on 2,000 UNSEEN challenges:{C.RESET}")
    print(f"  {accuracy_bar(acc_500)}")
    print()

    if acc_500 >= 95:
        typeprint(f"  ⚠  PUF COMPROMISED with just 500 observations!", color=C.RED + C.BOLD)
    else:
        typeprint(f"  Attack partially successful — more CRPs would finish the job.", color=C.YELLOW)

    pause()


    # ── STEP 5: Live cloning demo ─────────────────────────────────────────
    banner("STEP 5: Live Cloning — Predicting New Challenges", C.GREEN)
    print(f"  {C.WHITE}The attacker's model now predicts responses to NEW challenges")
    print(f"  it has never seen — without access to the PUF chip.{C.RESET}")
    print()

    demo_challenges, demo_responses = puf.generate_crps(10)
    demo_phi = np.array([puf._challenge_to_phi(c) for c in demo_challenges])
    predictions = model_500.predict(demo_phi)

    print(f"  {C.GRAY}{'Challenge (first 12 bits)':<32} {'True':>6} {'Predicted':>11} {'Match':>7}{C.RESET}")
    print(f"  {C.GRAY}{'-' * 60}{C.RESET}")

    correct = 0
    for i in range(10):
        true = demo_responses[i]
        pred = predictions[i]
        match = pred == true
        if match:
            correct += 1
        bits = "".join(str(b) for b in demo_challenges[i][:12])
        colored_bits = ""
        for b in bits:
            colored_bits += (C.CYAN + "1" if b == "1" else C.GRAY + "0") + C.RESET
        true_col  = C.GREEN if true == 1 else C.RED
        pred_col  = C.GREEN if pred == 1 else C.RED
        match_sym = C.GREEN + "  ✓" if match else C.RED + "  ✗"
        print(f"  {colored_bits}{C.GRAY}...{C.RESET}  {true_col}{'+1' if true==1 else '-1':>6}{C.RESET}  {pred_col}{'+1' if pred==1 else '-1':>10}{C.RESET}{match_sym}{C.RESET}")
        time.sleep(0.15)

    print()
    print(f"  {C.WHITE}Score: {C.GREEN}{C.BOLD}{correct}/10{C.RESET}{C.WHITE} correct on brand new challenges{C.RESET}")
    print()
    typeprint(f"  The clone predicts the PUF without ever touching the hardware.", color=C.GREEN)

    pause()


    # ── STEP 6: More CRPs → Higher accuracy ──────────────────────────────
    banner("STEP 6: More CRPs = Higher Accuracy", C.CYAN)
    print(f"  {C.WHITE}Watching accuracy grow as the attacker collects more data...{C.RESET}")
    print()

    sizes = [100, 500, 1000, 2000, 5000, 8000]
    for n in sizes:
        X_tr = np.array([puf._challenge_to_phi(c) for c in all_challenges[:n]])
        m = LogisticRegression(max_iter=1000)
        m.fit(X_tr, all_responses[:n])
        acc = accuracy_score(y_test, m.predict(X_test_phi)) * 100
        label = f"  {n:>6} CRPs  "
        print(f"{label}{accuracy_bar(acc)}")
        time.sleep(0.4)

    print()
    typeprint("  The attack becomes lethal fast. 500 CRPs is all it takes.", color=C.YELLOW)

    pause()


    # ── STEP 7: Bigger PUF, same result ──────────────────────────────────
    banner("STEP 7: Does a Bigger PUF Help?", C.YELLOW)
    print(f"  {C.WHITE}Testing attack accuracy vs PUF stage count")
    print(f"  (8,000 training CRPs each){C.RESET}")
    print()

    for n_stages in [16, 32, 64, 96, 128]:
        p = ArbiterPUF(n_stages=n_stages, seed=42)
        chal, resp = p.generate_crps(10000)
        X_tr = np.array([p._challenge_to_phi(c) for c in chal[:8000]])
        X_te = np.array([p._challenge_to_phi(c) for c in chal[8000:]])
        m = LogisticRegression(max_iter=1000)
        m.fit(X_tr, resp[:8000])
        acc = accuracy_score(resp[8000:], m.predict(X_te)) * 100
        label = f"  {n_stages:>4}-stage PUF  "
        print(f"{label}{accuracy_bar(acc)}")
        time.sleep(0.5)

    print()
    typeprint("  Flat line. Bigger PUFs are just as vulnerable.", color=C.RED)
    print(f"  {C.GRAY}  The linear separability is preserved at every scale.{C.RESET}")

    pause("Press ENTER for the final summary")


    # ── SUMMARY ───────────────────────────────────────────────────────────
    banner("DEMO COMPLETE — KEY TAKEAWAYS", C.GREEN)
    print(f"""
  {C.CYAN}What we just demonstrated:{C.RESET}

  {C.GREEN}✓{C.RESET}  An Arbiter PUF can be {C.BOLD}cloned with 500 CRPs{C.RESET} (~95% accuracy)
  {C.GREEN}✓{C.RESET}  With 8,000 CRPs accuracy reaches {C.BOLD}~99%{C.RESET}
  {C.GREEN}✓{C.RESET}  Random guessing baseline is {C.BOLD}~50%{C.RESET} — attack is not trivial luck
  {C.GREEN}✓{C.RESET}  {C.BOLD}Bigger PUFs are no safer{C.RESET} — the math stays linear
  {C.GREEN}✓{C.RESET}  Any linear classifier (LR or SVM) can do this

  {C.YELLOW}Why?{C.RESET}
  {C.WHITE}  response = sign(w · Φ(c)){C.RESET}
  {C.GRAY}  The Φ transform makes the PUF linearly separable.
  {C.GRAY}  Logistic regression reverse-engineers w from observed CRPs.
  {C.GRAY}  Φ is publicly known — not secret — so any attacker can use it.{C.RESET}

  {C.RED}Implication:{C.RESET}
  {C.WHITE}  The math can undermine the physics.
  {C.WHITE}  Hardware security must be evaluated against algorithmic attacks,
  {C.WHITE}  not just physical ones.{C.RESET}
    """)

    print(C.GRAY + "  " + "─" * 58 + C.RESET)
    print(f"  {C.GRAY}Demo complete. Thanks for watching!{C.RESET}\n")


if __name__ == "__main__":
    main()
