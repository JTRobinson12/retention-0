# Retention Model
This project models monthly termination risk for field leadership (superintendents and project managers) at a construction general contractor. It is a portfolio adaptation of an internal project: the pipeline, feature engineering, and model are real; the data is simulated so no company data leaves the building.

## Problem
Superintendents and project managers run job sites. When one leaves mid-build, the project loses continuity, a replacement takes months to source, and site relationships reset. Attrition among these positions is costly, and it is high throughout the industry, not just at any one company.

Two questions drive the model:
1. Which employees are at elevated risk of leaving in a given month?
2. Which working conditions relate to that risk, and are any of them things the business controls (project mix, relocation)?

The second question matters more than the first. A risk score identifies who to talk to; the conditions behind the score suggest what to change.

## Data
Three simulated sources mimic the structure of the original HR and project systems.

### People
One row per employee as of the extract date: age, sex, ethnicity, role, rehire status, home city, employment start and end dates, and a `terminated` flag. Employees still active at the extract date have no end date.

### Projects
One row per project: start and finish dates, city, `budget`, and `priority_project`, a flag for the company's most strategically important work. Larger projects run longer; priority projects are much larger and run moderately longer.

### Assignments
One row per stint of an employee on a project, with start and end dates. Field staff work one project at a time, separated by short bench gaps between assignments.

### Format
The modelling dataset is an employee-month panel built from the three sources. `terminated` is true only in an employee's final month, so each row asks: did this person leave this month? That makes the classifier a discrete-time hazard model, and the monthly positive rate is about 2.7 %. The panel is subset to superintendents and project managers because attrition among these positions is the costly one.

Inputs and why they are included:
* tenure (time varying): months since hire, in years. Attrition hazard typically falls as tenure accumulates.
* age (time varying): age in years at each month.
* `relocated`: the employee's current project is not in their home city. Time away from home is a leading candidate for why field staff quit.
* `budget`: project size. Bigger projects mean more staff, more incidents, and more pressure. Missing during bench months; the model handles missing values natively.
* `priority_project`: whether the current project is flagged priority. Staffing choices on marquee work may signal investment in the employee.
* `employee_percent_on_project`: the share of the project's elapsed life the employee has been present for. People who join a project late enter teams with defined relationships and subcultures, which can be difficult.
* months on project (time varying): how long the employee has been on their current assignment.
* `on_project`: whether the employee is assigned at all that month. Bench time may read as a signal about future work.
* male, ethnicity: included so the model can be audited for demographic effects, not because a demographic effect is desired.
* rehire: rehires have outside experience and options; no strong theory about direction.

### Simulation Design
Because the data is simulated, the true attrition process is known, which turns model assessment into a recovery test. Planted effects:
* Employees hired young leave at higher monthly rates.
* Rehires leave at slightly lower rates.
* Employees within six months of leaving are less likely to be staffed on priority projects and more likely to be placed out of town.

Sex, ethnicity, and everything else are noise by construction. A correct model recovers the planted effects in its SHAP values and assigns the noise features near-zero importance; the demographic columns double as a fairness check.

## Model
An XGBoost classifier estimates the probability of termination for each employee-month.

Choices:
* Gradient boosted trees fit the problem: mixed numeric and categorical inputs, nonlinear age and tenure effects, interactions between employee and project context, and native handling of missing values (project features during bench months).
* Class imbalance is handled with `scale_pos_weight` set to the negative-to-positive ratio, rather than resampling.
* Hyperparameters (depth, subsampling) come from Bayesian search with time-series cross-validation on the training window, so tuning never peeks at later months.
* The number of trees comes from early stopping against a holdout slice instead of being tuned directly.
* The train/validation/test split is 80/10/10 in time order. The deployment task is scoring future months, so evaluation is on months the model has never seen.
* SHAP values explain each score. One-hot encoded categories are recombined into single features so ethnicity reads as one effect rather than five columns.

## Running
```
uv sync
python recipes/simulate_data.py      # regenerate data/ (deterministic seeds)
python recipes/training/terminated_xgb.py
python recipes/scoring/terminated_xgb.py
```
Run from the repository root with `lib` on `PYTHONPATH`. Scored outputs and cached datasets land in `artifacts/`.
