"""Simulate the raw data sources for the retention model.

Each generator plants a small number of known effects (e.g., younger employees
leave at higher rates) so the downstream model has real signal to recover;
every other attribute is noise by construction.
"""

import numpy as np
import pandas as pd

# Fixing the "data pull" date makes generation fully deterministic.
from assets import EXTRACT_DATE

HIRING_WINDOW_START = pd.Timestamp("2015-01-01")

BASE_MONTHLY_ATTRITION = 0.025
DAYS_PER_MONTH = 30.44

HOME_LOCATIONS = {
    "St. Louis, MO": 0.30,
    "Chicago, IL": 0.20,
    "Atlanta, GA": 0.12,
    "Dallas, TX": 0.12,
    "Phoenix, AZ": 0.10,
    "Columbus, OH": 0.08,
    "Denver, CO": 0.08,
}

ETHNICITIES = {
    "White": 0.62,
    "Hispanic or Latino": 0.16,
    "Black or African American": 0.12,
    "Asian": 0.06,
    "Two or More Races": 0.04,
}

ROLES = {
    "Superintendent": 0.35,
    "Project Manager": 0.30,
    "Project Engineer": 0.20,
    "Safety Manager": 0.15,
}


def _weighted_choice(rng: np.random.Generator, options: dict[str, float], n: int) -> np.ndarray:
    """Sample n values from the keys of options, weighted by its values."""
    return rng.choice(list(options), size=n, p=list(options.values()))


def projects(n_projects: int = 180, seed: int = 17) -> pd.DataFrame:
    """Simulate the portfolio of construction projects.

    A latent size factor drives both budget and duration so that bigger
    projects cost more and run longer. Priority projects (e.g., mission
    critical work) carry much larger budgets and run moderately longer.
    Projects still underway at EXTRACT_DATE keep their planned end date.
    """
    rng = np.random.default_rng(seed)

    priority_project = rng.random(n_projects) < 0.12
    size_factor = rng.lognormal(0, 0.6, n_projects)

    budget = 8e6 * size_factor**1.5 * np.where(priority_project, 6, 1)
    duration_months = (12 * size_factor * np.where(priority_project, 1.5, 1)).clip(4, 60)

    start_window_days = (EXTRACT_DATE - HIRING_WINDOW_START).days
    project_start = HIRING_WINDOW_START + pd.to_timedelta(rng.integers(0, start_window_days, n_projects), unit="D")
    project_end = project_start + pd.to_timedelta(np.round(duration_months * DAYS_PER_MONTH), unit="D")

    return pd.DataFrame({
        "project_number": np.arange(20001, 20001 + n_projects),
        "project_start": project_start,
        "project_end": project_end,
        "location": _weighted_choice(rng, HOME_LOCATIONS, n_projects),
        "priority_project": priority_project,
        "budget": (budget / 1e5).round() * 1e5,
    })


def people(n_people: int = 2000, seed: int = 13) -> pd.DataFrame:
    """Simulate the employee roster as of EXTRACT_DATE.

    Termination is drawn from a monthly geometric hazard with two planted
    effects: employees hired young leave more often, and rehires leave less
    often. Age is drawn at hire and aged forward to EXTRACT_DATE so age and
    tenure stay consistent. Employees whose drawn exit lands after
    EXTRACT_DATE are still active (terminated is False and employment_end is
    missing).
    """
    rng = np.random.default_rng(seed)

    age_at_hire = rng.normal(36, 9, n_people).clip(21, 60).round(0)
    rehire = rng.random(n_people) < 0.08

    hiring_window_days = (EXTRACT_DATE - HIRING_WINDOW_START).days
    employment_start = HIRING_WINDOW_START + pd.to_timedelta(rng.integers(0, hiring_window_days, n_people), unit="D")
    age = (age_at_hire + (EXTRACT_DATE - employment_start).days / 365.25).round(0)

    monthly_hazard = (BASE_MONTHLY_ATTRITION * np.exp((40 - age_at_hire) / 25) * np.where(rehire, 0.8, 1.0)).clip(
        0.004, 0.25
    )
    duration_months = rng.geometric(monthly_hazard)
    exit_date = employment_start + pd.to_timedelta(np.round(duration_months * DAYS_PER_MONTH), unit="D")

    terminated = exit_date <= EXTRACT_DATE
    employment_end = exit_date.where(terminated, pd.NaT)
    tenure_end = exit_date.where(terminated, EXTRACT_DATE)

    return pd.DataFrame({
        "employee_number": np.arange(10001, 10001 + n_people),
        "terminated": terminated,
        "age": age,
        "tenure_years": ((tenure_end - employment_start).days / 365.25).round(2),
        "male": rng.random(n_people) < 0.72,
        "ethnicity": _weighted_choice(rng, ETHNICITIES, n_people),
        "role": _weighted_choice(rng, ROLES, n_people),
        "rehire": rehire,
        "home_location": _weighted_choice(rng, HOME_LOCATIONS, n_people),
        "employment_start": employment_start,
        "employment_end": employment_end,
    })


NEAR_EXIT = pd.Timedelta(days=180)


def _stint_weights(candidates: pd.DataFrame, home_location: str, near_exit: bool) -> np.ndarray:
    """Weight candidate projects for one stint.

    Everyone prefers projects near home. People close to their exit date are
    steered away from priority projects and toward out-of-town work, planting
    the association between project mix and termination that the model
    should recover.
    """
    weights = np.where(candidates.location == home_location, 3.0, 1.0)
    if near_exit:
        weights = weights * np.where(candidates.priority_project, 0.25, 1.0)
        weights = weights * np.where(candidates.location == home_location, 1.0, 2.5)
    return weights / weights.sum()


def assignments(people_df: pd.DataFrame, projects_df: pd.DataFrame, seed: int = 19) -> pd.DataFrame:
    """Simulate project assignments by walking each person's employment timeline.

    Each person works consecutive single-project stints separated by short
    bench gaps. Stints respect both the person's employment window and the
    project's date range.
    """
    rng = np.random.default_rng(seed)

    rows = []
    for person in people_df.itertuples():
        exit_date = person.employment_end if person.terminated else EXTRACT_DATE
        current = person.employment_start + pd.Timedelta(days=int(rng.integers(0, 21)))
        while current < exit_date:
            candidates = projects_df.query("project_start <= @current < project_end")
            if candidates.empty:
                current += pd.Timedelta(days=30)
                continue
            near_exit = person.terminated and (exit_date - current) < NEAR_EXIT
            project = candidates.iloc[
                rng.choice(len(candidates), p=_stint_weights(candidates, person.home_location, near_exit))
            ]
            stint = pd.Timedelta(days=round(rng.lognormal(np.log(8), 0.5) * DAYS_PER_MONTH))
            assignment_end = min(project.project_end, exit_date, current + stint)
            rows.append((project.project_number, person.employee_number, current, assignment_end))
            current = assignment_end + pd.Timedelta(days=int(rng.integers(5, 45)))

    return pd.DataFrame(rows, columns=["project_number", "employee_number", "assignment_start", "assignment_end"])
