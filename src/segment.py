"""Unsupervised customer segmentation with K-Means.

Run from the project root:  python -m src.segment
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data_prep import ROOT, TARGET, load_dataset

SEED = 42
FEATURES = ["tenure", "MonthlyCharges", "NumServices"]
FIG_DIR = ROOT / "reports" / "figures"


def choose_k(X, k_range=range(2, 9)):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def name_segment(row, med):
    tenure = "Loyal" if row["tenure"] >= med["tenure"] else "New"
    spend = "high-spend" if row["MonthlyCharges"] >= med["MonthlyCharges"] else "low-spend"
    return f"{tenure}, {spend}"


def main(k=4):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    X = StandardScaler().fit_transform(df[FEATURES])

    scores = choose_k(X)
    print(scores.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(scores["k"], scores["inertia"], marker="o"); axes[0].set_title("Elbow method")
    axes[1].plot(scores["k"], scores["silhouette"], marker="o", color="#c0504d")
    axes[1].set_title("Silhouette score")
    for ax in axes:
        ax.set_xlabel("k")
    fig.tight_layout(); fig.savefig(FIG_DIR / "choose_k.png", dpi=150); plt.close(fig)

    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    df["Segment"] = km.labels_
    profile = df.groupby("Segment").agg(
        customers=(TARGET, "size"), tenure=("tenure", "mean"),
        MonthlyCharges=("MonthlyCharges", "mean"), NumServices=("NumServices", "mean"),
        churn_rate=(TARGET, "mean"),
        month_to_month=("Contract", lambda s: (s == "Month-to-month").mean()),
    ).round(2)
    med = df[FEATURES].median()
    profile["label"] = profile.apply(lambda r: name_segment(r, med), axis=1)
    profile = profile.sort_values("churn_rate", ascending=False)
    print("\nSegment profiles:\n", profile.to_string())

    pcs = PCA(n_components=2, random_state=SEED).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    for seg in sorted(df["Segment"].unique()):
        m = df["Segment"] == seg
        ax.scatter(pcs[m, 0], pcs[m, 1], s=6, alpha=0.5,
                   label=f"{seg}: {profile.loc[seg, 'label']} ({profile.loc[seg, 'churn_rate']:.0%} churn)")
    ax.set_title("Customer segments (PCA projection)"); ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "segments.png", dpi=150); plt.close(fig)

    (ROOT / "reports" / "segments.json").write_text(
        json.dumps(profile.reset_index().to_dict(orient="records"), indent=2))
    return profile


if __name__ == "__main__":
    main()
