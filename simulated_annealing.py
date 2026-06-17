"""
Répartition de deals entre 3 brokers (buckets) - Optimisation multi-critères
Critères : delta, DV01, valeur de marché (continus) + devises (catégoriel)
Algorithme : Simulated Annealing
"""

import pandas as pd
import numpy as np
import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────

def load_deals(filepath: str, sheet_name: str = 0) -> pd.DataFrame:
    """
    Charge les deals depuis un fichier Excel.

    Colonnes attendues (noms personnalisables via COLUMN_MAP ci-dessous) :
        deal_id   : identifiant unique du deal
        delta     : sensibilité delta (float)
        dv01      : DV01 en devise (float)
        market_value : valeur de marché (float)
        currency  : code devise ISO (str, ex: 'EUR', 'USD', 'GBP')
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df


# ── Adapter ces noms aux colonnes réelles de votre fichier Excel ──
COLUMN_MAP = {
    "deal_id":      "deal_id",       # identifiant du deal
    "delta":        "delta",          # sensibilité delta
    "dv01":         "dv01",           # DV01
    "market_value": "market_value",   # valeur de marché
    "currency":     "currency",       # devise (catégoriel)
}

# Poids de chaque critère dans la fonction objectif (doivent sommer à 1.0)
WEIGHTS = {
    "delta":        0.30,
    "dv01":         0.30,
    "market_value": 0.25,
    "currency":     0.15,
}

N_BROKERS = 3
BROKER_NAMES = ["Broker A", "Broker B", "Broker C"]


# ─────────────────────────────────────────────
# 2. GÉNÉRATION DE DONNÉES FICTIVES (démo)
# ─────────────────────────────────────────────

def generate_sample_deals(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Génère un DataFrame de deals fictifs pour tester l'algorithme."""
    rng = np.random.default_rng(seed)
    currencies = rng.choice(["EUR", "USD", "GBP", "JPY", "CHF"], size=n,
                             p=[0.35, 0.30, 0.15, 0.12, 0.08])
    return pd.DataFrame({
        "deal_id":      [f"D{i:04d}" for i in range(n)],
        "delta":        rng.uniform(-500_000, 500_000, n).round(0),
        "dv01":         rng.uniform(-50_000,   50_000, n).round(2),
        "market_value": rng.uniform(-10_000_000, 10_000_000, n).round(0),
        "currency":     currencies,
    })


# ─────────────────────────────────────────────
# 3. FONCTION OBJECTIF
# ─────────────────────────────────────────────

def compute_cost(assignment: list[int], df: pd.DataFrame,
                 weights: dict, scales: dict) -> float:
    """
    Calcule le coût de la répartition courante.

    Critères continus (delta, dv01, market_value) :
        variance entre buckets des sommes, normalisée par l'échelle.

    Critère catégoriel (currency) :
        pour chaque devise, variance entre buckets du % de notionnel,
        moyennée sur toutes les devises présentes.

    Parameters
    ----------
    assignment : liste de longueur n_deals, assignment[i] ∈ {0,1,2}
    df         : DataFrame des deals (index aligné sur assignment)
    weights    : dict des poids par critère
    scales     : dict des facteurs de normalisation (std de chaque critère)
    """
    assignment = np.array(assignment)
    cost = 0.0

    # ── Critères continus ──
    for col in ["delta", "dv01", "market_value"]:
        bucket_sums = np.array([
            df[col].values[assignment == k].sum()
            for k in range(N_BROKERS)
        ])
        # variance des sommes, normalisée
        var = np.var(bucket_sums) / (scales[col] ** 2 + 1e-12)
        cost += weights[col] * var

    # ── Critère catégoriel : devises ──
    currencies = df["currency"].values
    unique_ccys = np.unique(currencies)
    ccy_costs = []
    for ccy in unique_ccys:
        mask = currencies == ccy
        counts = np.array([mask[assignment == k].sum() for k in range(N_BROKERS)])
        total = counts.sum()
        if total == 0:
            continue
        fractions = counts / total          # part de cette devise dans chaque bucket
        ccy_costs.append(np.var(fractions))
    if ccy_costs:
        cost += weights["currency"] * np.mean(ccy_costs)

    return cost


def compute_scales(df: pd.DataFrame) -> dict:
    """Calcule les facteurs de normalisation (std × sqrt(n_brokers))."""
    n = len(df)
    expected_per_bucket = n / N_BROKERS
    return {
        col: abs(df[col].std() * np.sqrt(expected_per_bucket))
        for col in ["delta", "dv01", "market_value"]
    }


# ─────────────────────────────────────────────
# 4. SIMULATED ANNEALING
# ─────────────────────────────────────────────

def simulated_annealing(
    df: pd.DataFrame,
    weights: dict,
    T0: float = 1.0,
    T_min: float = 1e-5,
    cooling_rate: float = 0.995,
    max_iter: int = 100_000,
    seed: int = 0,
) -> tuple[list[int], float, list[float]]:
    """
    Simulated Annealing pour minimiser le déséquilibre multi-critères.

    À chaque itération :
      - on choisit un deal aléatoire
      - on le réassigne à un broker différent
      - on accepte si amélioration, ou avec probabilité exp(-Δcost/T)

    Returns
    -------
    best_assignment : liste d'entiers (broker 0/1/2 pour chaque deal)
    best_cost       : coût final
    cost_history    : historique du coût (pour visualisation)
    """
    random.seed(seed)
    np.random.seed(seed)

    n = len(df)
    scales = compute_scales(df)

    # Initialisation aléatoire (équilibrée)
    assignment = [i % N_BROKERS for i in range(n)]
    random.shuffle(assignment)

    current_cost = compute_cost(assignment, df, weights, scales)
    best_assignment = assignment[:]
    best_cost = current_cost

    T = T0
    cost_history = [current_cost]

    for iteration in range(max_iter):
        # Perturbation : déplacer un deal vers un autre broker
        idx = random.randrange(n)
        old_broker = assignment[idx]
        new_broker = random.choice([b for b in range(N_BROKERS) if b != old_broker])

        assignment[idx] = new_broker
        new_cost = compute_cost(assignment, df, weights, scales)
        delta = new_cost - current_cost

        # Critère d'acceptation de Metropolis
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_assignment = assignment[:]
        else:
            assignment[idx] = old_broker  # annuler le mouvement

        T *= cooling_rate
        if T < T_min:
            break

        if iteration % 5000 == 0:
            cost_history.append(current_cost)

    return best_assignment, best_cost, cost_history


# ─────────────────────────────────────────────
# 5. RÉSULTATS & AFFICHAGE
# ─────────────────────────────────────────────

def build_result_df(df: pd.DataFrame, assignment: list[int]) -> pd.DataFrame:
    result = df.copy()
    result["broker"] = [BROKER_NAMES[a] for a in assignment]
    return result


def print_summary(result: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("RÉSUMÉ PAR BROKER")
    print("=" * 60)

    summary = result.groupby("broker").agg(
        n_deals=("deal_id", "count"),
        sum_delta=("delta", "sum"),
        sum_dv01=("dv01", "sum"),
        sum_mv=("market_value", "sum"),
    ).round(2)
    print(summary.to_string())

    print("\n── Distribution des devises par broker ──")
    ccy_table = result.groupby(["broker", "currency"]).size().unstack(fill_value=0)
    print(ccy_table.to_string())

    # Déséquilibres
    print("\n── Déséquilibres (max - min entre brokers) ──")
    for col, label in [("sum_delta", "Delta"), ("sum_dv01", "DV01"), ("sum_mv", "MV")]:
        spread = summary[col].max() - summary[col].min()
        print(f"  {label:20s}: {spread:,.2f}")


def plot_results(result: pd.DataFrame, cost_history: list[float]) -> None:
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Répartition deals entre brokers — Optimisation multi-critères",
                 fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = ["#185FA5", "#3B6D11", "#854F0B"]
    brokers = BROKER_NAMES

    # ── 1. Delta par broker ──
    ax1 = fig.add_subplot(gs[0, 0])
    vals = [result[result["broker"] == b]["delta"].sum() for b in brokers]
    ax1.bar(brokers, vals, color=colors)
    ax1.set_title("Somme Delta", fontsize=11)
    ax1.set_ylabel("Delta")
    ax1.axhline(0, color="gray", linewidth=0.7)
    ax1.tick_params(axis="x", labelsize=9)

    # ── 2. DV01 par broker ──
    ax2 = fig.add_subplot(gs[0, 1])
    vals = [result[result["broker"] == b]["dv01"].sum() for b in brokers]
    ax2.bar(brokers, vals, color=colors)
    ax2.set_title("Somme DV01", fontsize=11)
    ax2.axhline(0, color="gray", linewidth=0.7)
    ax2.tick_params(axis="x", labelsize=9)

    # ── 3. Valeur de marché par broker ──
    ax3 = fig.add_subplot(gs[0, 2])
    vals = [result[result["broker"] == b]["market_value"].sum() for b in brokers]
    ax3.bar(brokers, vals, color=colors)
    ax3.set_title("Valeur de marché", fontsize=11)
    ax3.axhline(0, color="gray", linewidth=0.7)
    ax3.tick_params(axis="x", labelsize=9)

    # ── 4. Distribution devises (stacked bar) ──
    ax4 = fig.add_subplot(gs[1, 0:2])
    ccy_table = result.groupby(["broker", "currency"]).size().unstack(fill_value=0)
    ccy_table.plot(kind="bar", ax=ax4, legend=True)
    ax4.set_title("Nombre de deals par devise et broker", fontsize=11)
    ax4.set_xlabel("")
    ax4.tick_params(axis="x", rotation=0, labelsize=9)
    ax4.legend(title="Devise", fontsize=9, title_fontsize=9)

    # ── 5. Convergence du coût ──
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(cost_history, color="#185FA5", linewidth=1.2)
    ax5.set_title("Convergence (coût objectif)", fontsize=11)
    ax5.set_xlabel("Itération (×5000)")
    ax5.set_ylabel("Coût")

    plt.savefig("deal_partitioning_results.png", dpi=150, bbox_inches="tight")
    print("\n→ Graphique sauvegardé : deal_partitioning_results.png")
    plt.show()


# ─────────────────────────────────────────────
# 6. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run(filepath: str | None = None, sheet_name: str = 0) -> pd.DataFrame:
    """
    Point d'entrée principal.

    Parameters
    ----------
    filepath   : chemin vers le fichier Excel. Si None → données fictives.
    sheet_name : nom ou index de la feuille Excel (défaut : première feuille).

    Exemple d'utilisation :
        result = run("deals.xlsx", sheet_name="Sheet1")
        result.to_excel("deals_avec_brokers.xlsx", index=False)
    """
    # Chargement
    if filepath:
        print(f"Chargement des données depuis : {filepath}")
        df = load_deals(filepath, sheet_name=sheet_name)
        # Renommer les colonnes selon COLUMN_MAP si nécessaire
        df = df.rename(columns={v: k for k, v in COLUMN_MAP.items()})
    else:
        print("⚠ Aucun fichier fourni — utilisation de données fictives (60 deals)")
        df = generate_sample_deals(n=60)

    print(f"  {len(df)} deals chargés | Devises : {sorted(df['currency'].unique())}")

    # Optimisation
    print("\nOptimisation en cours (Simulated Annealing)…")
    assignment, best_cost, cost_history = simulated_annealing(
        df,
        weights=WEIGHTS,
        T0=1.0,
        cooling_rate=0.995,
        max_iter=100_000,
        seed=42,
    )
    print(f"  Coût final : {best_cost:.6f}")

    # Résultats
    result = build_result_df(df, assignment)
    print_summary(result)
    plot_results(result, cost_history)

    # Export
    out_path = "deals_avec_brokers.xlsx"
    result.to_excel(out_path, index=False)
    print(f"\n→ Fichier exporté : {out_path}")

    return result


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Option 1 : données fictives (démo) ──
    result = run()

    # ── Option 2 : votre fichier Excel ──
    # result = run("chemin/vers/votre_fichier.xlsx", sheet_name="Feuille1")
