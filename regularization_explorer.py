import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ZERO_TOL = 1e-6
C_VALUES = np.logspace(-3, 2, 20)  # from 0.001 to 100


def parse_args():
    parser = argparse.ArgumentParser(description="Regularization Explorer for Logistic Regression")
    parser.add_argument("csv_path", help="Path to the telecom churn CSV file")
    parser.add_argument(
        "--target",
        default=None,
        help="Target column name. If omitted, the script will try to infer it.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where plots/csv/interpretation will be saved",
    )
    return parser.parse_args()


def infer_target_column(df: pd.DataFrame, explicit_target: str | None = None) -> str:
    if explicit_target is not None:
        if explicit_target not in df.columns:
            raise ValueError(f"Target column '{explicit_target}' was not found in the dataset.")
        return explicit_target

    candidates = [
        "churned",
        "Churn",
        "churn",
        "Exited",
        "target",
        "Target",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not infer the target column automatically. "
        "Pass it manually with --target"
    )


def encode_binary_target(y: pd.Series) -> pd.Series:
    if y.nunique(dropna=True) != 2:
        raise ValueError("Target must be binary for LogisticRegression.")

    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)

    y_clean = y.astype(str).str.strip().str.lower()

    positive_values = {"yes", "true", "1", "churn", "churned"}
    negative_values = {"no", "false", "0", "stay", "not churned"}

    unique_vals = set(y_clean.unique())

    if unique_vals.issubset(positive_values.union(negative_values)):
        mapping = {}
        for v in unique_vals:
            if v in positive_values:
                mapping[v] = 1
            elif v in negative_values:
                mapping[v] = 0
        return y_clean.map(mapping).astype(int)

    # fallback: categorical codes for any 2-class string target
    cat = pd.Categorical(y_clean)
    encoded = pd.Series(cat.codes, index=y.index)

    if encoded.nunique() != 2:
        raise ValueError("Failed to convert target to binary values.")

    print("Target mapping used:")
    for code, label in enumerate(cat.categories):
        print(f"  {label} -> {code}")

    return encoded.astype(int)


def preprocess_data(df: pd.DataFrame, target_col: str):
    df = df.copy()

    # Drop obvious ID-like columns if present
    drop_if_present = {"customerid", "customer_id", "id"}
    cols_to_drop = [c for c in df.columns if c.lower() in drop_if_present]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    y = encode_binary_target(df[target_col])
    X = df.drop(columns=[target_col])

    # Clean strings
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype(str).str.strip()

    # Fill missing values
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            mode_value = X[col].mode().iloc[0] if not X[col].mode().empty else "missing"
            X[col] = X[col].fillna(mode_value)

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X, drop_first=True, dtype=float)

    feature_names = X_encoded.columns.tolist()

    # Standardize so coefficients are comparable
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    return X_encoded, X_scaled, y, feature_names


def fit_regularization_paths(X_scaled, y, penalty: str, c_values):
    paths = []

    for c in c_values:
        model = LogisticRegression(
            penalty=penalty,
            C=c,
            solver="liblinear",   # supports both l1 and l2 for binary classification
            max_iter=5000,
            random_state=42,
        )
        model.fit(X_scaled, y)
        paths.append(model.coef_.ravel())

    return np.array(paths)


def get_highlight_features(l1_paths, l2_paths, feature_names, top_k=8):
    # Highlight features that are strongest overall across both penalties
    combined_strength = np.max(np.abs(l1_paths), axis=0) + np.max(np.abs(l2_paths), axis=0)
    top_idx = np.argsort(combined_strength)[-top_k:][::-1]
    return [feature_names[i] for i in top_idx]


def compute_l1_entry_points(l1_paths, feature_names, c_values, zero_tol=ZERO_TOL):
    """
    For each feature, find the first C (moving from strong reg -> weak reg)
    where the coefficient becomes non-zero.
    A larger entry C means the feature needs weaker regularization to survive,
    so it is eliminated earlier when regularization strengthens.
    """
    records = []

    for j, feature in enumerate(feature_names):
        non_zero_idx = np.where(np.abs(l1_paths[:, j]) > zero_tol)[0]

        if len(non_zero_idx) == 0:
            # zero across all tested C values
            records.append(
                {
                    "feature": feature,
                    "entry_idx": None,
                    "entry_C": None,
                    "always_zero": True,
                }
            )
        else:
            idx = non_zero_idx[0]
            records.append(
                {
                    "feature": feature,
                    "entry_idx": int(idx),
                    "entry_C": float(c_values[idx]),
                    "always_zero": False,
                }
            )

    # Features zeroed out earliest under strengthening regularization
    # are those with the largest entry_C, excluding ones active even at smallest C
    eliminated_first = [
        r for r in records
        if (r["entry_idx"] is not None and r["entry_idx"] > 0)
    ]
    eliminated_first = sorted(eliminated_first, key=lambda x: x["entry_C"], reverse=True)

    always_zero = [r for r in records if r["always_zero"]]

    return records, eliminated_first, always_zero


def build_interpretation(feature_names, l1_paths, l2_paths, c_values):
    _, eliminated_first, always_zero = compute_l1_entry_points(l1_paths, feature_names, c_values)

    # robust features: large average absolute magnitude under L2
    mean_abs_l2 = np.mean(np.abs(l2_paths), axis=0)
    robust_idx = np.argsort(mean_abs_l2)[-3:][::-1]
    robust_features = [feature_names[i] for i in robust_idx]

    early_zero_features = [r["feature"] for r in eliminated_first[:3]]
    if not early_zero_features and always_zero:
        early_zero_features = [r["feature"] for r in always_zero[:3]]

    # simple recommendation rule
    mid_idx = len(c_values) // 2
    l1_zero_fraction_mid = np.mean(np.abs(l1_paths[mid_idx]) <= ZERO_TOL)

    if l1_zero_fraction_mid >= 0.30:
        recommendation = (
            "Based on this path, L1 is a strong choice if you want a simpler and more interpretable model, "
            "because it performs useful feature selection by driving weaker predictors to zero."
        )
    else:
        recommendation = (
            "Based on this path, L2 is the safer production choice, because it shrinks coefficients smoothly "
            "while preserving information across correlated predictors instead of aggressively removing them."
        )

    early_zero_text = ", ".join(early_zero_features) if early_zero_features else "some weaker predictors"
    robust_text = ", ".join(robust_features) if robust_features else "the strongest predictors"

    paragraph = (
        f"The regularization paths show a clear difference between L1 and L2 on this telecom churn dataset. "
        f"As regularization becomes stronger (smaller C), L1 pushes several coefficients exactly to zero, "
        f"with {early_zero_text} disappearing earliest, which suggests that these features are weaker or more redundant "
        f"once other predictors are present. In contrast, {robust_text} remain comparatively more stable across the "
        f"regularization spectrum, making them likely to be the most robust churn predictors in the dataset. "
        f"L2 shrinks coefficients continuously toward zero but rarely eliminates them completely, so it preserves more signal "
        f"from correlated features while reducing variance. {recommendation}"
    )

    return paragraph


def save_coefficient_csv(paths, feature_names, c_values, output_path):
    coef_df = pd.DataFrame(paths, columns=feature_names)
    coef_df.insert(0, "C", c_values)
    coef_df.to_csv(output_path, index=False)


def plot_regularization_paths(
    feature_names,
    l1_paths,
    l2_paths,
    c_values,
    output_path,
):
    highlight_features = set(get_highlight_features(l1_paths, l2_paths, feature_names, top_k=8))
    _, eliminated_first, always_zero = compute_l1_entry_points(l1_paths, feature_names, c_values)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    panel_data = [
        ("L1 (Lasso) Regularization Path", l1_paths, axes[0]),
        ("L2 (Ridge) Regularization Path", l2_paths, axes[1]),
    ]

    for title, paths, ax in panel_data:
        for j, feature in enumerate(feature_names):
            is_highlighted = feature in highlight_features
            ax.plot(
                c_values,
                paths[:, j],
                linewidth=2.0 if is_highlighted else 1.0,
                alpha=0.95 if is_highlighted else 0.15,
                label=feature if is_highlighted else None,
            )

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("C (log scale)")
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Coefficient value")

    # Annotate L1 early-zero features
    top_annotated = eliminated_first[:5]
    if not top_annotated and always_zero:
        top_annotated = always_zero[:5]

    summary_lines = ["Zeroed out earliest under L1:"]
    for i, record in enumerate(top_annotated, start=1):
        feature = record["feature"]

        if record.get("entry_C") is not None:
            entry_c = record["entry_C"]
            entry_idx = record["entry_idx"]
            j = feature_names.index(feature)
            y_val = l1_paths[entry_idx, j]

            axes[0].scatter([entry_c], [y_val], s=35, zorder=5)
            axes[0].annotate(
                feature,
                xy=(entry_c, y_val),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
            summary_lines.append(f"{i}. {feature} (appears at C={entry_c:.3g})")
        else:
            summary_lines.append(f"{i}. {feature} (zero across all tested C values)")

    axes[0].text(
        0.02,
        0.02,
        "\n".join(summary_lines),
        transform=axes[0].transAxes,
        fontsize=9,
        va="bottom",
        bbox=dict(boxstyle="round", alpha=0.85),
    )

    axes[1].text(
        0.02,
        0.02,
        "L2 shrinks coefficients toward zero\nbut usually does not make them exactly zero.",
        transform=axes[1].transAxes,
        fontsize=9,
        va="bottom",
        bbox=dict(boxstyle="round", alpha=0.85),
    )

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            title="Highlighted features",
        )

    plt.tight_layout(rect=(0, 0, 0.87, 1))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    target_col = infer_target_column(df, args.target)

    print(f"Detected target column: {target_col}")

    X_encoded, X_scaled, y, feature_names = preprocess_data(df, target_col)

    print(f"Rows: {len(df)}")
    print(f"Encoded feature count: {len(feature_names)}")

    l1_paths = fit_regularization_paths(X_scaled, y, penalty="l1", c_values=C_VALUES)
    l2_paths = fit_regularization_paths(X_scaled, y, penalty="l2", c_values=C_VALUES)

    save_coefficient_csv(
        l1_paths,
        feature_names,
        C_VALUES,
        output_dir / "l1_coefficients.csv",
    )
    save_coefficient_csv(
        l2_paths,
        feature_names,
        C_VALUES,
        output_dir / "l2_coefficients.csv",
    )

    plot_regularization_paths(
        feature_names,
        l1_paths,
        l2_paths,
        C_VALUES,
        output_dir / "regularization_path.png",
    )

    interpretation = build_interpretation(feature_names, l1_paths, l2_paths, C_VALUES)
    with open(output_dir / "interpretation.md", "w", encoding="utf-8") as f:
        f.write(interpretation + "\n")

    print("\nDone. Files saved to:", output_dir.resolve())
    print("- regularization_path.png")
    print("- l1_coefficients.csv")
    print("- l2_coefficients.csv")
    print("- interpretation.md")


if __name__ == "__main__":
    main()