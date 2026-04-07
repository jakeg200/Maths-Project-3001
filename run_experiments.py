import json, random, time, os, re, argparse
import numpy as np
from collections import defaultdict


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# 5 models: 1 OpenAI (paid), 4 Groq (free)
# Clean parameter ladder: 8B → 32B → 70B → 120B → frontier
# 3 providers: Meta, Alibaba, OpenAI
MODELS = {
    "gpt-5.4": {
        "provider": "openai",
        "model_id": "gpt-5.4",
        "label": "GPT-5.4",
        "params": "Frontier (undisclosed)",
    },
    "gpt-oss-120b": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
        "label": "GPT-OSS 120B",
        "params": "120B",
    },
    "llama-70b": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "label": "Llama 3.3 70B",
        "params": "70B",
    },
    "llama-8b": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "label": "Llama 3.1 8B",
        "params": "8B",
    },
    "qwen-32b": {
        "provider": "groq",
        "model_id": "qwen/qwen3-32b",
        "label": "Qwen 3 32B",
        "params": "32B",
    },
}

SYSTEM_PROMPT = "Answer the following question."
TEMPERATURE = 0
N_SURFACE = 50
N_STRUCTURAL = 50  # 15 flip + 13 mediator + 12 reverse + 10 numerical


# SURFACE MUTATION WORD BANKS (20 domains)


DOMAINS = [
    {"confounder": "neighbourhood poverty level", "treatment": "government funding", "outcome": "dropout rates",
     "context": "A government study examined schools across the country."},
    {"confounder": "patient illness severity", "treatment": "medication dosage", "outcome": "hospital stay length",
     "context": "A hospital reviewed patient records from the past year."},
    {"confounder": "crime rate", "treatment": "police officers deployed", "outcome": "arrests made",
     "context": "A city council analysed policing data across districts."},
    {"confounder": "soil contamination", "treatment": "fertiliser applied", "outcome": "crop failure rate",
     "context": "An agricultural study examined farms in a drought-prone region."},
    {"confounder": "traffic congestion", "treatment": "traffic officers deployed", "outcome": "accidents reported",
     "context": "A transport department reviewed accident data."},
    {"confounder": "building deterioration", "treatment": "maintenance budget", "outcome": "tenant complaints",
     "context": "A housing authority examined records across its properties."},
    {"confounder": "student prior attainment", "treatment": "tutoring hours", "outcome": "exam failure rate",
     "context": "A university examined its tutoring programme."},
    {"confounder": "population density", "treatment": "doctors per capita", "outcome": "disease prevalence",
     "context": "A public health study compared regions."},
    {"confounder": "river pollution level", "treatment": "water treatment investment", "outcome": "fish mortality",
     "context": "An environmental agency studied waterways."},
    {"confounder": "unemployment rate", "treatment": "welfare spending", "outcome": "homelessness rate",
     "context": "A social policy institute examined 200 local authorities."},
    {"confounder": "wildfire risk level", "treatment": "rangers stationed", "outcome": "forest hectares burned",
     "context": "A national park service reviewed decade-long data."},
    {"confounder": "earthquake magnitude", "treatment": "rescue teams deployed", "outcome": "casualties",
     "context": "A disaster response agency analysed recent earthquakes."},
    {"confounder": "storm severity", "treatment": "emergency responders", "outcome": "infrastructure damage",
     "context": "A government report examined major storm responses."},
    {"confounder": "company debt level", "treatment": "consultants hired", "outcome": "layoffs",
     "context": "A business school studied corporate restructuring."},
    {"confounder": "software bug severity", "treatment": "QA engineers assigned", "outcome": "customer complaints",
     "context": "A tech company reviewed product quality data."},
    {"confounder": "legal case complexity", "treatment": "lawyers assigned", "outcome": "trial duration",
     "context": "A law firm analysed its case portfolio."},
    {"confounder": "infection severity", "treatment": "antibiotic dosage", "outcome": "recovery time",
     "context": "A clinical trial examined bacterial infection treatments."},
    {"confounder": "road surface deterioration", "treatment": "repair crew hours", "outcome": "pothole complaints",
     "context": "A highways agency reviewed maintenance data."},
    {"confounder": "child behavioural difficulty", "treatment": "specialist support hours", "outcome": "school exclusions",
     "context": "An education authority examined outcomes for children with additional needs."},
    {"confounder": "machine wear level", "treatment": "maintenance hours", "outcome": "production downtime",
     "context": "A factory analysed equipment maintenance records."},
]

MEDIATOR_NAMES = [
    "immune response", "chemical reaction", "secondary process",
    "behavioural change", "inflammatory response", "feedback mechanism",
    "regulatory cascade", "compensatory mechanism",
]

# PROMPT GENERATORS


def make_l1():
    return {
        "id": "L1_original", "level": 1, "mutation_type": "none",
        "prompt": (
            "A city analysed records of past fires and found that fires where "
            "more firefighters were dispatched suffered more property damage. "
            "The city council proposes cutting the fire department budget to "
            "reduce property damage.\n\n"
            "Is the council's reasoning valid? Answer Yes or No, then explain."
        ),
        "correct_binary": "no",
        "confounder_name": "severity",
        "reasoning_terms": ["severity", "confound", "common cause",
                            "spurious", "hidden", "third variable"],
    }

def make_l2(domain, idx):
    conf_key = domain["confounder"].split()[-1].lower()
    return {
        "id": f"L2_{idx:03d}", "level": 2, "mutation_type": "surface",
        "prompt": (
            f"{domain['context']} The data shows that areas with more "
            f"{domain['treatment']} tend to have higher {domain['outcome']}. "
            f"A decision-maker proposes reducing {domain['treatment']} to "
            f"reduce {domain['outcome']}.\n\n"
            f"Is the decision-maker's reasoning valid? Answer Yes or No, "
            f"then explain."
        ),
        "correct_binary": "no",
        "confounder_name": conf_key,
        "reasoning_terms": [conf_key, domain["confounder"].split()[0].lower(),
                            "confound", "common cause", "spurious",
                            "hidden", "driven by"],
    }

def make_l3_flip(domain, idx):
    return {
        "id": f"L3_flip_{idx:03d}", "level": 3, "mutation_type": "flip_sign",
        "prompt": (
            f"{domain['context']} The data shows that areas with more "
            f"{domain['treatment']} tend to have higher {domain['outcome']}. "
            f"The correlation is confounded by {domain['confounder']}, which "
            f"drives both. However, unlike typical cases, {domain['treatment']} "
            f"actually makes {domain['outcome']} worse due to an adverse "
            f"interaction.\n\n"
            f"If we deliberately increase {domain['treatment']}, will "
            f"{domain['outcome']} increase or decrease? Answer Increase or "
            f"Decrease, then explain."
        ),
        "correct_binary": "increase",
        "confounder_name": domain["confounder"].split()[-1].lower(),
        "reasoning_terms": ["adverse", "worse", "harmful", "makes things worse",
                            "amplif", "exacerbat", "increase damage",
                            "increase " + domain["outcome"].split()[0].lower()],
    }

def make_l3_med(domain, idx):
    med = random.choice(MEDIATOR_NAMES)
    med_key = med.split()[0]
    return {
        "id": f"L3_med_{idx:03d}", "level": 3, "mutation_type": "add_mediator",
        "mediator": med,
        "prompt": (
            f"{domain['context']} {domain['treatment'].capitalize()} does not "
            f"affect {domain['outcome']} directly. Instead, it triggers a "
            f"{med}, which in turn reduces {domain['outcome']}. "
            f"{domain['confounder'].capitalize()} independently drives both "
            f"{domain['treatment']} and {domain['outcome']}.\n\n"
            f"Consider a specific case where {domain['confounder']} was high, "
            f"{domain['treatment']} was provided, and {domain['outcome']} was "
            f"moderate. Had {domain['treatment']} NOT been provided in this "
            f"specific case, would {domain['outcome']} have been worse? "
            f"Answer Yes or No, then explain your reasoning for this "
            f"specific case."
        ),
        "correct_binary": "yes",
        "confounder_name": domain["confounder"].split()[-1].lower(),
        "reasoning_terms": ["this specific", "this case", "this particular",
                            "individual", "counterfactual", "had not",
                            "would have", med_key],
    }

def make_l3_rev(domain, idx):
    return {
        "id": f"L3_rev_{idx:03d}", "level": 3, "mutation_type": "reverse_edge",
        "prompt": (
            f"{domain['context']} The data shows that areas with more "
            f"{domain['treatment']} tend to have higher {domain['outcome']}. "
            f"However, {domain['treatment']} does not cause {domain['outcome']} "
            f"--- it merely detects or reveals pre-existing {domain['outcome']} "
            f"that would have occurred regardless.\n\n"
            f"Would reducing {domain['treatment']} reduce the actual amount of "
            f"{domain['outcome']}? Answer Yes or No, then explain."
        ),
        "correct_binary": "no",
        "confounder_name": domain["confounder"].split()[-1].lower(),
        "reasoning_terms": ["detect", "reveal", "does not cause",
                            "not the same as caus", "would still",
                            "regardless", "observation"],
    }

def make_l3_num(idx):
    a = random.choice([1, 2, 3])
    b = random.choice([2, 3, 4])
    g = random.choice([1, 2])
    sv = sorted(random.sample(range(1, 6), 3))
    S = np.array(sv, dtype=float)
    F = a * S
    D = b * S - g * F
    fv = random.choice([2, 4, 6])
    correct_cov = float(np.cov(F, D, ddof=0)[0, 1])
    correct_int = float(b * np.mean(S) - g * fv)
    return {
        "id": f"L3_num_{idx:03d}", "level": 3, "mutation_type": "numerical",
        "prompt": (
            f"Consider a system with three variables S, F, and D:\n"
            f"  S = U_S (exogenous, takes values {{{sv[0]}, {sv[1]}, "
            f"{sv[2]}}} equally)\n"
            f"  F = {a}S\n"
            f"  D = {b}S - {g}F\n\n"
            f"Q1: What is Cov(F, D)?\n"
            f"Q2: If we intervene to set F = {fv} (removing the equation "
            f"F = {a}S), what is E[D | do(F = {fv})]?\n\n"
            f"Give numerical answers."
        ),
        "correct_binary": None,
        "correct_cov": correct_cov,
        "correct_int": correct_int,
        "confounder_name": None,
        "reasoning_terms": ["do(", "intervene", "remov", "sever",
                            "structural equation"],
    }

def generate_all():
    """Generate all 101 problems with fixed seed."""
    random.seed(42)
    np.random.seed(42)
    probs = [make_l1()]

    for i, d in enumerate(random.choices(DOMAINS, k=N_SURFACE)):
        probs.append(make_l2(d, i))

    ds = random.choices(DOMAINS, k=40)
    for i in range(15):
        probs.append(make_l3_flip(ds[i], i))
    for i in range(13):
        probs.append(make_l3_med(ds[15 + i], i))
    for i in range(12):
        probs.append(make_l3_rev(ds[28 + i], i))
    for i in range(10):
        probs.append(make_l3_num(i))

    return probs


def preview_problems(probs):
    """Print example problems for inspection (Spyder-friendly)."""
    n1 = sum(1 for p in probs if p["level"] == 1)
    n2 = sum(1 for p in probs if p["level"] == 2)
    n3 = sum(1 for p in probs if p["level"] == 3)
    print(f"Generated {len(probs)} problems: L1={n1}, L2={n2}, L3={n3}")
    n_flip = sum(1 for p in probs if p.get("mutation_type") == "flip_sign")
    n_med = sum(1 for p in probs if p.get("mutation_type") == "add_mediator")
    n_rev = sum(1 for p in probs if p.get("mutation_type") == "reverse_edge")
    n_num = sum(1 for p in probs if p.get("mutation_type") == "numerical")
    print(f"  L3: flip={n_flip}, med={n_med}, rev={n_rev}, num={n_num}")

    for label, filt in [("L1", lambda p: p["level"] == 1),
                         ("L2", lambda p: p["level"] == 2),
                         ("L3 flip", lambda p: p.get("mutation_type") == "flip_sign"),
                         ("L3 mediator", lambda p: p.get("mutation_type") == "add_mediator"),
                         ("L3 reverse", lambda p: p.get("mutation_type") == "reverse_edge"),
                         ("L3 numerical", lambda p: p.get("mutation_type") == "numerical")]:
        ex = next((p for p in probs if filt(p)), None)
        if ex:
            print(f"\n--- {label}: {ex['id']} ---")
            print(ex["prompt"][:300])
            print(f"Correct: {ex.get('correct_binary', 'numerical')}")



# AUTOMATED SCORING (two binary signals)


def strip_think_tags(response):
    """Remove <redacted_thinking>...</redacted_thinking> blocks that Qwen 3 and other
    reasoning models prepend to their answers."""
    return re.sub(r'<redacted_thinking>.*?</redacted_thinking>', '', response, flags=re.DOTALL).strip()


def extract_answer(response):
    """Extract Yes/No/Increase/Decrease from response.
    Handles models that use <redacted_thinking> blocks before answering."""
    # Strip thinking blocks first
    clean = strip_think_tags(response)
    if not clean:
        # Fallback: search the whole response
        clean = response

    first = clean.lower().split('\n')[0].strip()
    words = first.split()
    if not words:
        return "unclear"

    # Check first word
    fw = re.sub(r'[^a-z]', '', words[0])
    if fw in ["yes", "no", "increase", "decrease"]:
        return fw

    # Check first line for keywords
    for kw in ["yes", "no", "increase", "decrease"]:
        if kw in first:
            return kw

    # Last resort: check full cleaned response for the FIRST occurrence
    for kw in ["no", "yes", "increase", "decrease"]:
        if kw in clean.lower()[:500]:
            return kw

    return "unclear"


def score_correctness(problem, response):
    """Binary correctness: 1 if answer matches ground truth, 0 otherwise."""
    if problem.get("correct_binary") is None:
        # Numerical problem
        clean = strip_think_tags(response)
        nums = [float(n) for n in re.findall(r'-?\d+\.?\d*', clean)]
        cov_ok = any(abs(n - problem["correct_cov"]) < 0.5 for n in nums)
        int_ok = any(abs(n - problem["correct_int"]) < 0.5 for n in nums)
        return 1 if (cov_ok and int_ok) else 0
    else:
        ans = extract_answer(response)
        return 1 if ans == problem["correct_binary"].lower() else 0


def score_reasoning(problem, response):
    """Binary reasoning: 1 if response mentions >= 2 relevant causal terms.
    Searches full response INCLUDING think blocks — reasoning in <redacted_thinking>
    is still reasoning."""
    resp_lower = response.lower()
    terms = problem.get("reasoning_terms", [])
    if not terms:
        return 0
    matches = sum(1 for t in terms if t.lower() in resp_lower)
    threshold = min(2, max(1, len(terms) // 3))
    return 1 if matches >= threshold else 0


def auto_score(problem, response):
    """Return dict with both scores."""
    c = score_correctness(problem, response)
    r = score_reasoning(problem, response)
    return {"correctness": c, "reasoning": r, "total": c + r}



# API CALLS (2 providers: OpenAI + Groq)


def _require_api_keys(provider: str):
    if provider == "openai" and not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or add it to a .env file "
            "(not committed); see README."
        )
    if provider == "groq" and not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it or add it to a .env file "
            "(not committed); see README."
        )


def call_openai(prompt, model_id, system=SYSTEM_PROMPT):
    _require_api_keys("openai")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=TEMPERATURE, max_completion_tokens=1000,
    )
    return resp.choices[0].message.content


def call_groq(prompt, model_id, system=SYSTEM_PROMPT):
    _require_api_keys("groq")
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=TEMPERATURE, max_tokens=1000,
    )
    return resp.choices[0].message.content


def call_model(prompt, model_key, system=SYSTEM_PROMPT):
    """Route to correct provider."""
    cfg = MODELS[model_key]
    p = cfg["provider"]
    mid = cfg["model_id"]
    if p == "openai":
        return call_openai(prompt, mid, system)
    elif p == "groq":
        return call_groq(prompt, mid, system)
    else:
        raise ValueError(f"Unknown provider: {p}")



# MULTI-AGENT DEBATE (GPT-5 only, on 20 L3 problems)


def run_debate(problem, model_key="gpt-5.4", rounds=3):
    """Three-round debate: Agent N (naive) vs Agent S (structural)."""
    sys_n = ("You believe the observational data is reliable. Argue that the "
             "correlation reflects a genuine causal relationship. Challenge "
             "any claims about confounding or hidden variables. Be concise.")
    sys_s = ("You believe the observational data is misleading. Argue that "
             "there is a confounding variable or structural explanation. "
             "Use causal reasoning to support your position. Be concise.")

    hist_n, hist_s = [], []
    for r in range(rounds):
        if r == 0:
            msg_n = (f"{problem['prompt']}\n\nArgue that the observed "
                     f"pattern reflects the true causal effect.")
        else:
            msg_n = (f"Your opponent argued:\n{hist_s[-1]}\n\n"
                     f"Respond and defend your position.")
        resp_n = call_model(msg_n, model_key, sys_n)
        hist_n.append(resp_n)
        time.sleep(1)

        msg_s = (f"Your opponent argued:\n{resp_n}\n\n"
                 f"Respond and defend your causal analysis position.")
        resp_s = call_model(msg_s, model_key, sys_s)
        hist_s.append(resp_s)
        time.sleep(1)

    # Judge (no biased system prompt)
    judge_msg = (
        f"Two analysts debated this problem:\n\n{problem['prompt']}\n\n"
        f"Analyst A's final argument:\n{hist_n[-1]}\n\n"
        f"Analyst B's final argument:\n{hist_s[-1]}\n\n"
        f"Based on both arguments, provide your final answer to the "
        f"original question."
    )
    judge_resp = call_model(judge_msg, model_key)
    sc = auto_score(problem, judge_resp)

    return {
        "problem_id": problem["id"], "model": model_key,
        "agent_n_final": hist_n[-1], "agent_s_final": hist_s[-1],
        "judge_response": judge_resp, **sc,
    }



# RUN ALL EXPERIMENTS


def run_all_experiments(probs, save_every=10):
    """Run all models on all problems. Saves incrementally."""
    results = []
    total = len(probs) * len(MODELS)
    done = 0

    for mk in MODELS:
        print(f"\n{'='*50}")
        print(f"  {MODELS[mk]['label']} ({MODELS[mk]['params']})")
        print(f"{'='*50}")
        for i, p in enumerate(probs):
            done += 1
            print(f"  [{done}/{total}] {p['id']}", end=" ", flush=True)
            try:
                resp = call_model(p["prompt"], mk)
                sc = auto_score(p, resp)
                results.append({
                    "problem_id": p["id"], "level": p["level"],
                    "mutation_type": p.get("mutation_type", "none"),
                    "model": mk, "response": resp, **sc,
                })
                print(f"C={sc['correctness']} R={sc['reasoning']}")

                # Rate limit: Groq free tier is 30 RPM for most models
                if MODELS[mk]["provider"] == "groq":
                    time.sleep(2.5)  # ~24 requests/min, safe margin
                else:
                    time.sleep(0.5)

            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "problem_id": p["id"], "level": p["level"],
                    "mutation_type": p.get("mutation_type", "none"),
                    "model": mk, "response": str(e),
                    "correctness": 0, "reasoning": 0, "total": 0,
                })
                # If rate limited, wait longer
                if "rate" in str(e).lower():
                    print("    Rate limited — waiting 60s...")
                    time.sleep(60)

            # Save incrementally
            if done % save_every == 0:
                with open("all_results_partial.json", "w") as f:
                    json.dump(results, f, indent=2)

    # Final save
    with open("all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to all_results.json")
    return results



# STATISTICS AND OUTPUT


def compute_and_print(results):
    """Print everything needed for the dissertation."""
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["model"], r["level"])].append(r)

    print("\n" + "=" * 70)
    print("  DISSERTATION RESULTS")
    print("=" * 70)

    # ---- Main table ----
    print("\nTABLE: Correctness (%) and Reasoning (%) by model and level\n")
    print(f"{'Model':20s} | {'L1 C':>5} {'L1 R':>5} | {'L2 C':>5} {'L2 R':>5} | {'L3 C':>5} {'L3 R':>5}")
    print("-" * 70)
    for mk in MODELS:
        lab = MODELS[mk]["label"]
        row = f"{lab:20s} |"
        for lv in [1, 2, 3]:
            recs = grouped.get((mk, lv), [])
            if recs:
                corr = np.mean([r["correctness"] for r in recs]) * 100
                reas = np.mean([r["reasoning"] for r in recs]) * 100
                row += f" {corr:5.0f} {reas:5.0f} |"
            else:
                row += "   ---   --- |"
        print(row)

    # ---- Overall means ----
    print("\nOVERALL MEANS:")
    for lv in [1, 2, 3]:
        recs = [r for r in results if r["level"] == lv]
        if recs:
            c = np.mean([r["correctness"] for r in recs]) * 100
            r_ = np.mean([r["reasoning"] for r in recs]) * 100
            print(f"  L{lv}: Corr={c:.1f}%, Reas={r_:.1f}% (n={len(recs)})")


    # ---- L3 by mutation type ----
    print("\nL3 BY MUTATION TYPE:")
    for mt in ["flip_sign", "add_mediator", "reverse_edge", "numerical"]:
        sub = [r for r in results if r.get("mutation_type") == mt]
        if sub:
            c = np.mean([r["correctness"] for r in sub]) * 100
            r_ = np.mean([r["reasoning"] for r in sub]) * 100
            print(f"  {mt:15s}: Corr={c:.0f}%, Reas={r_:.0f}% (n={len(sub)})")

    # ---- L3 by mutation type per model ----
    print("\nL3 CORRECTNESS BY MUTATION TYPE (per model):")
    print(f"{'Model':20s} | {'flip':>6} {'med':>6} {'rev':>6} {'num':>6}")
    print("-" * 55)
    for mk in MODELS:
        lab = MODELS[mk]["label"]
        row = f"{lab:20s} |"
        for mt in ["flip_sign", "add_mediator", "reverse_edge", "numerical"]:
            sub = [r for r in results if r["model"] == mk and r.get("mutation_type") == mt]
            if sub:
                c = np.mean([r["correctness"] for r in sub]) * 100
                row += f" {c:5.0f}%"
            else:
                row += "   ---"
        print(row)

    # ---- Correctness-Reasoning gap ----
    print("\nCORRECTNESS vs REASONING GAP:")
    for lv_name, lv in [("L2", 2), ("L3", 3)]:
        print(f"\n  {lv_name}:")
        for mk in MODELS:
            recs = [r for r in results if r["model"] == mk and r["level"] == lv]
            if recs:
                c = np.mean([r["correctness"] for r in recs]) * 100
                r_ = np.mean([r["reasoning"] for r in recs]) * 100
                gap = c - r_
                print(f"    {MODELS[mk]['label']:20s}: Corr={c:5.0f}%, "
                      f"Reas={r_:5.0f}%, Gap={gap:+5.0f}%")


def generate_charts(results):
    """Write matplotlib figures under figures/ from scored result rows."""
    if not results:
        print("No results to chart.")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping charts.")
        return

    os.makedirs("figures", exist_ok=True)
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["model"], r["level"])].append(r["correctness"])

    model_keys = list(MODELS.keys())
    labels = [MODELS[k]["label"] for k in model_keys]
    x = np.arange(len(model_keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, lv in enumerate([1, 2, 3]):
        means = []
        for mk in model_keys:
            recs = grouped.get((mk, lv), [])
            means.append(float(np.mean(recs)) if recs else 0.0)
        ax.bar(x + (i - 1) * width, means, width, label=f"L{lv}")

    ax.set_ylabel("Correctness rate")
    ax.set_title("Correctness by model and difficulty level")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "correctness_by_level.png"), dpi=150)
    plt.close(fig)

    # L3 mutation breakdown (all models pooled)
    mut_types = ["flip_sign", "add_mediator", "reverse_edge", "numerical"]
    sub_means = []
    for mt in mut_types:
        recs = [r["correctness"] for r in results if r.get("mutation_type") == mt]
        sub_means.append(float(np.mean(recs)) if recs else 0.0)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(mut_types, sub_means, color=["#4472c4", "#ed7d31", "#70ad47", "#ffc000"])
    ax2.set_ylabel("Correctness rate")
    ax2.set_title("L3 correctness by mutation type (all models)")
    ax2.set_ylim(0, 1.05)
    fig2.tight_layout()
    fig2.savefig(os.path.join("figures", "l3_by_mutation.png"), dpi=150)
    plt.close(fig2)
    print("Wrote figures/correctness_by_level.png and figures/l3_by_mutation.png")



# MAIN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutations", action="store_true",
                        help="Generate and preview problems only")
    parser.add_argument("--stats", action="store_true",
                        help="Recompute stats from all_results.json")
    parser.add_argument("--debate", action="store_true",
                        help="Run debate only (needs all_results.json)")
    parser.add_argument("--charts", action="store_true",
                        help="Generate charts from all_results.json")
    args = parser.parse_args()

    probs = generate_all()

    if args.mutations:
        preview_problems(probs)
        return

    if args.stats:
        with open("all_results.json") as f:
            results = json.load(f)
        compute_and_print(results)
        generate_charts(results)
        return

    if args.charts:
        with open("all_results.json") as f:
            results = json.load(f)
        generate_charts(results)
        return

    # Full pipeline
    if not args.debate:
        results = run_all_experiments(probs)
        compute_and_print(results)
        generate_charts(results)

    # Debate (GPT-5 only on 20 L3 problems)
    print("\n\nMULTI-AGENT DEBATE (GPT-5.4, 20 L3 problems)...")
    l3_subset = [p for p in probs if p["level"] == 3][:20]
    debate_results = []

    for p in l3_subset:
        print(f"  {p['id']}", end=" ", flush=True)
        try:
            r = run_debate(p, "gpt-5.4")
            debate_results.append(r)
            print(f"C={r['correctness']} R={r['reasoning']}")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(1)

    with open("debate_results.json", "w") as f:
        json.dump(debate_results, f, indent=2)

    # Compare
    if os.path.exists("all_results.json"):
        with open("all_results.json") as f:
            main_res = json.load(f)
    else:
        main_res = results if 'results' in dir() else []

    single = [r["correctness"] for r in main_res
              if r["model"] == "gpt-5.4" and r["level"] == 3]
    debate = [r["correctness"] for r in debate_results]
    s_pct = np.mean(single) * 100 if single else 0
    d_pct = np.mean(debate) * 100 if debate else 0
    print(f"\nDEBATE RESULT: single={s_pct:.0f}%, debate={d_pct:.0f}%, "
          f"delta={d_pct - s_pct:+.0f}%")


if __name__ == "__main__":
    main()
