import io
import math
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats


st.set_page_config(page_title="A/B Test Lab", layout="wide")
st.title("A/B Test Lab")
st.caption(
    "Upload your data, choose the metric type, and run the appropriate test "
    "to see whether Variant B performs differently from Variant A."
)


def load_data(uploaded_file) -> pd.DataFrame:
    """Read a CSV file into a DataFrame while trying a few common encodings."""
    if uploaded_file is None:
        return pd.DataFrame()

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding, engine="python")

    uploaded_file.seek(0)
    raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
    return pd.read_csv(io.StringIO(raw_text))


@st.cache_data(show_spinner=False)
def load_sample_data(seed: int = 55) -> pd.DataFrame:
    """Generate a small synthetic dataset covering numeric, binary, and categorical metrics."""
    rng = np.random.default_rng(seed)

    n_obs = 400
    groups = rng.choice(["Variant A", "Variant B"], size=n_obs, p=[0.5, 0.5])
    base_metric = rng.normal(loc=50, scale=10, size=n_obs)
    lift = np.where(groups == "Variant B", 4.0, 0.0)
    revenue = base_metric + lift + rng.normal(scale=5, size=n_obs)

    conversion_prob = np.where(groups == "Variant B", 0.62, 0.55)
    converted = rng.binomial(1, conversion_prob)

    support_topics = np.where(
        groups == "Variant B",
        rng.choice(["General", "Technical", "Other"], p=[0.4, 0.45, 0.15], size=n_obs),
        rng.choice(["General", "Technical", "Other"], p=[0.5, 0.35, 0.15], size=n_obs),
    )

    return pd.DataFrame(
        {
            "user_id": np.arange(1, n_obs + 1),
            "group": groups,
            "revenue": revenue.round(2),
            "converted": converted,
            "support_topic": support_topics,
        }
    )


def infer_metric_type(series: pd.Series) -> str:
    series_no_na = series.dropna()
    if series_no_na.empty:
        return "numeric"

    if pd.api.types.is_numeric_dtype(series_no_na):
        unique_values = np.unique(series_no_na)
        if set(np.round(unique_values).astype(int)).issubset({0, 1}) and len(unique_values) <= 2:
            return "binary"
        if series_no_na.nunique() <= 5:
            return "categorical"
        return "numeric"

    if series_no_na.nunique() <= 2:
        return "binary"
    return "categorical"


def render_group_selection(df: pd.DataFrame) -> Tuple[str, str, str]:
    group_candidates = [col for col in df.columns if df[col].nunique(dropna=True) >= 2]
    if not group_candidates:
        st.warning("No suitable grouping column found. Add a categorical column with at least two values.")
        return "", "", ""

    group_col = st.selectbox("Select the column that identifies the variants", group_candidates, index=0)
    if not group_col:
        return "", "", ""

    valid_groups = df[group_col].dropna().unique().tolist()
    if len(valid_groups) < 2:
        st.warning("The selected group column needs at least two distinct values.")
        return "", "", ""

    selected_groups = st.multiselect(
        "Pick the two variants you want to compare", valid_groups, default=valid_groups[:2]
    )

    if len(selected_groups) != 2:
        st.info("Select exactly two variants to continue.")
        return group_col, "", ""

    base_variant = st.selectbox(
        "Which variant should be treated as the baseline?",
        selected_groups,
        index=0,
    )
    test_variant = [g for g in selected_groups if g != base_variant][0]

    return group_col, base_variant, test_variant


def compute_numeric_test(a: pd.Series, b: pd.Series) -> dict:
    a_clean = pd.to_numeric(a, errors="coerce").dropna()
    b_clean = pd.to_numeric(b, errors="coerce").dropna()

    if len(a_clean) < 2 or len(b_clean) < 2:
        raise ValueError("Need at least two observations per group for a t-test.")

    test_result = stats.ttest_ind(a_clean, b_clean, equal_var=False)

    mean_a, mean_b = a_clean.mean(), b_clean.mean()
    std_a, std_b = a_clean.std(ddof=1), b_clean.std(ddof=1)

    n_a, n_b = len(a_clean), len(b_clean)
    mean_diff = mean_b - mean_a

    var_a = std_a ** 2
    var_b = std_b ** 2
    se_diff = math.sqrt(var_a / n_a + var_b / n_b)

    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a ** 2) / (n_a ** 2 * (n_a - 1)) + (var_b ** 2) / (n_b ** 2 * (n_b - 1))
    dof = df_num / df_den if df_den > 0 else max(min(n_a, n_b) - 1, 1)

    t_crit = stats.t.ppf(0.975, dof)
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    return {
        "test_name": "Welch's t-test",
        "statistic": test_result.statistic,
        "p_value": test_result.pvalue,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "std_a": std_a,
        "std_b": std_b,
        "n_a": n_a,
        "n_b": n_b,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def compute_binary_test(a: pd.Series, b: pd.Series, success_value) -> dict:
    a_success = (a == success_value).astype(int)
    b_success = (b == success_value).astype(int)

    n_a, n_b = len(a_success), len(b_success)
    x_a, x_b = a_success.sum(), b_success.sum()

    if n_a == 0 or n_b == 0:
        raise ValueError("Need observations in both variants to run the test.")

    p_a = x_a / n_a
    p_b = x_b / n_b
    diff = p_b - p_a

    pooled = (x_a + x_b) / (n_a + n_b)
    pooled_var = pooled * (1 - pooled) * (1 / n_a + 1 / n_b)

    if pooled_var == 0:
        raise ValueError("Not enough variability to run a z-test. Check the metric values.")

    z_score = diff / math.sqrt(pooled_var)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(0.975)
    ci_low = diff - z_crit * se
    ci_high = diff + z_crit * se

    return {
        "test_name": "Two-proportion z-test",
        "statistic": z_score,
        "p_value": p_value,
        "p_a": p_a,
        "p_b": p_b,
        "n_a": n_a,
        "n_b": n_b,
        "diff": diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "success_value": success_value,
    }


def compute_categorical_test(df: pd.DataFrame, group_col: str, metric_col: str) -> dict:
    contingency = pd.crosstab(df[group_col], df[metric_col])
    if contingency.shape[1] < 2:
        raise ValueError("Categorical metrics need at least two distinct values for the selected variants.")

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    total = contingency.values.sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = math.sqrt(chi2 / (total * min_dim)) if min_dim > 0 else np.nan

    return {
        "test_name": "Chi-square test of independence",
        "statistic": chi2,
        "p_value": p_value,
        "dof": dof,
        "cramers_v": cramers_v,
        "contingency": contingency,
        "expected": pd.DataFrame(expected, index=contingency.index, columns=contingency.columns),
    }


with st.sidebar:
    st.header("Upload or sample data")
    data_source = st.radio("Data source", ["Upload CSV", "Use sample dataset"], index=1)

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = load_data(uploaded_file)
            except Exception as exc:  # pragma: no cover - user input specific
                st.error(str(exc))
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
    else:
        df = load_sample_data()

    if not df.empty:
        st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        show_preview = st.checkbox("Show data preview", value=True)
        if show_preview:
            st.dataframe(df.head(20))

if df.empty:
    st.info("Start by uploading a CSV file or switch to the autogenerated sample dataset in the sidebar.")
    st.stop()

st.subheader("1. Choose the variants")
group_col, base_variant, test_variant = render_group_selection(df)
if not group_col or not base_variant or not test_variant:
    st.stop()

subset = df[df[group_col].isin([base_variant, test_variant])].copy()
subset[group_col] = subset[group_col].astype(str)

st.write(
    f"Comparing `{test_variant}` against `{base_variant}` using {len(subset)} observations."
)

st.subheader("2. Choose the metric to evaluate")
metric_options = [col for col in subset.columns if col != group_col]
if not metric_options:
    st.error("No metric columns available. Ensure your dataset has at least one metric column.")
    st.stop()

metric_col = st.selectbox("Metric column", metric_options)
if not metric_col:
    st.stop()

metric_series = subset[metric_col]

default_metric_type = infer_metric_type(metric_series)
metric_options_display = ["Auto detect", "Numeric", "Binary", "Categorical"]
default_index = 0
if default_metric_type == "numeric":
    default_index = 1
elif default_metric_type == "binary":
    default_index = 2
elif default_metric_type == "categorical":
    default_index = 3

metric_type = st.selectbox("Metric type", metric_options_display, index=default_index)
if metric_type == "Auto detect":
    metric_type = default_metric_type.capitalize()

metric_type = metric_type.lower()

a_values = subset[subset[group_col] == base_variant][metric_col]
b_values = subset[subset[group_col] == test_variant][metric_col]

st.subheader("3. Run the test")

try:
    if metric_type == "numeric":
        result = compute_numeric_test(a_values, b_values)
        st.markdown("### Result: Welch's t-test")

        cols = st.columns(2)
        cols[0].metric("Mean (Variant A)", f"{result['mean_a']:.3f}")
        cols[1].metric("Mean (Variant B)", f"{result['mean_b']:.3f}", f"{result['mean_diff']:.3f}")

        summary = f"t-statistic = {result['statistic']:.3f}, p-value = {result['p_value']:.4f}."
        if result["p_value"] < 0.05:
            summary += " The difference is statistically significant at the 5% level."
        st.write(summary)
        st.write(
            f"95% CI for (B - A): [{result['ci_low']:.3f}, {result['ci_high']:.3f}]."
        )
        st.write(
            f"Sample sizes: Variant A = {result['n_a']} (std = {result['std_a']:.3f}), "
            f"Variant B = {result['n_b']} (std = {result['std_b']:.3f})."
        )

    elif metric_type == "binary":
        unique_values = sorted(metric_series.dropna().unique().tolist())
        if not unique_values:
            raise ValueError("Metric column is empty after dropping missing values.")
        success_default = unique_values[-1]
        success_index = unique_values.index(success_default) if success_default in unique_values else 0
        success_value = st.selectbox("Which value counts as a success?", unique_values, index=success_index)

        result = compute_binary_test(a_values, b_values, success_value)
        st.markdown("### Result: Two-proportion z-test")

        cols = st.columns(2)
        cols[0].metric("Conversion (Variant A)", f"{result['p_a']:.2%}")
        cols[1].metric("Conversion (Variant B)", f"{result['p_b']:.2%}", f"{result['diff']:.2%}")

        summary = f"z-score = {result['statistic']:.3f}, p-value = {result['p_value']:.4f}."
        if result["p_value"] < 0.05:
            summary += " The difference is statistically significant at the 5% level."
        st.write(summary)
        st.write(
            f"95% CI for lift (B - A): [{result['ci_low']:.2%}, {result['ci_high']:.2%}]."
        )
        st.write(
            f"Samples: Variant A = {result['n_a']} rows, Variant B = {result['n_b']} rows."
        )

    elif metric_type == "categorical":
        result = compute_categorical_test(subset, group_col, metric_col)
        st.markdown("### Result: Chi-square test of independence")

        summary = (
            f"chi-square = {result['statistic']:.3f}, dof = {result['dof']}, p-value = {result['p_value']:.4f}."
        )
        if result["p_value"] < 0.05:
            summary += " The distribution differs significantly at the 5% level."
        st.write(summary)
        st.write(
            f"Cramer's V = {result['cramers_v']:.3f} (effect size; 0 = no association, 1 = perfect association)."
        )

        st.write("Observed counts:")
        st.dataframe(result['contingency'])

        st.write("Expected counts (if variants behaved the same):")
        st.dataframe(result['expected'].round(2))
    else:
        st.error("Unsupported metric type. Please choose Numeric, Binary, or Categorical.")
except ValueError as exc:  # pragma: no cover - runtime safeguard for user data
    st.error(str(exc))


st.subheader("4. Document your findings")
st.text_area(
    "Add experiment notes (optional)",
    placeholder="Summarise your key takeaways, business impact, next steps, and thoughts.",
    height=120,
)

st.success("All done! You can tweak the selections above or upload a new dataset to run another test.")

st.sidebar.markdown("---")
st.sidebar.markdown("Need help? Please visit https://github.com/dipika-desaboyina/abtestlab.\n")
st.sidebar.markdown("---")
st.sidebar.markdwon("Here is a sample Kaggle dataset you can use for testing - https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "- Metric must be measured per row.\n"
    "- Group column should identify the two variants.\n"
    "- For binary metrics, pick the value that counts as a success."
)

