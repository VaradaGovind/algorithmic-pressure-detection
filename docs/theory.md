# Theoretical framing and contributions

This document summarizes the theoretical contribution and assumptions used in the project.

- Stress measurement: physiological signals are converted into engineered features and aggregated into anomaly scores using an autoencoder trained on baseline data. We interpret higher reconstruction error as elevated physiological stress.
- Causal identification: to move beyond correlation we use a combination of (i) time-aggregated linkage with lag analysis and Granger causality tests, (ii) panel regression with subject fixed effects to control for time-invariant confounders, and (iii) propensity-score matching to adjust for observable covariates.
- Decision-theoretic integration: we translate stress-linked differences into expected economic costs and compare operational policies (buffer stock, flexible workforce) to quantify trade-offs.

Assumptions and limitations:
- Observational data: causal claims rely on the assumption that major confounders are observed or time-invariant.
- Aggregation: binning into time windows (default 5 minutes) trades temporal resolution for alignment with operational KPIs.
- External validity: cross-dataset evaluation and simulated external cohorts are included to probe generalizability.

Recommended next steps for theoretical strengthening:
- Use instrumental variables or natural experiments if available in production data.
- Formalize a structural model linking stress to throughput and quality for counterfactual policy evaluation.
