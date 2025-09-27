# A/B Test Lab

A Streamlit dashboard for running quick A/B tests on numeric, binary, and categorical metrics. Upload experiment data or explore the bundled sample dataset, select the right metric type, and review statistical results with confidence intervals and effect sizes.

**Live demo:** https://abtestlab.streamlit.app/

## Features
- Load data from CSV or use an auto-generated sample dataset for demos.
- Auto-detect the metric type with the option to override.
- Variant picker that compares any two groups in your data.
- Appropriate statistical tests for each metric:
  - Welch's t-test for numeric outcomes
  - Two-proportion z-test for binary metrics
  - Chi-square test (with Cramer's V) for categorical distributions
- Inline explanations, data previews, and a notes panel to capture findings.

## Try It Your Way
- **Use the hosted version** when you need a quick walkthrough or want to share results: [abtestlab.streamlit.app](https://abtestlab.streamlit.app/).
- **Run it locally** if you prefer to work offline, analyze sensitive datasets, or tweak the code.

## Local Requirements
- Python 3.9+
- pip packages: `streamlit`, `pandas`, `numpy`, `scipy`

> The repository includes a local virtual environment in `dev_env/`; you can ignore it if you prefer creating your own.

## Local Quick Start
1. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```bash
   streamlit run abtest.py
   ```

## Using the App
1. **Upload or Sample Data** – Use the sidebar to upload a CSV or explore the sample dataset.
2. **Choose Variants** – Pick the column that identifies experiment variants and select the two groups to compare.
3. **Pick a Metric** – Select the metric column; keep auto-detection or set the metric type manually.
4. **Run the Test** – Review the appropriate statistical test, p-values, confidence intervals, and effect sizes.
5. **Document Findings** – Capture experiment notes directly in the app for future reference.

## Data Expectations
- Each row should represent an observation (e.g., a user or session).
- Include a column that labels the experiment variant (e.g., `group`).
- Numeric metrics should be continuous values; binary metrics should have two distinct values; categorical metrics can have multiple categories.

## Troubleshooting
- Empty previews or errors typically mean the CSV could not be parsed—check delimiter, encoding, or missing headers.
- Ensure both variants have data and the metric column has at least two distinct values to run the selected test.

## Next Steps
- Export results or notes for sharing with stakeholders.
- Extend the app with additional statistical tests or multiple-metric comparisons.
