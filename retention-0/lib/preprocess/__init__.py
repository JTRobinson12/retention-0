import pandas as pd
from functionals import pandas_extensions as pdx
from plum import ReadFromCache, storage

import assets
import keys
import piping
from preprocess import assignments, people, projects, splitter

MERGE_INSTRUCTIONS = (
    (assignments.process, {"how": "left", "on": ["employee_number", "relative_month"]}),
    (projects.process, {"how": "left", "on": ["project_number", "relative_month"]}),
)


def process() -> pd.DataFrame:
    """Merge the person, assignment, and project panels and prepare data for modelling.

    Employee-months without an assignment are bench months: they have no
    project attributes, zero time on project, and are not relocated.
    """
    return (
        people.process()
        .pipe(pdx.reduce_merge, ((process(), merge_kwargs) for process, merge_kwargs in MERGE_INSTRUCTIONS))
        .assign(
            on_project=lambda _df: _df.project_number.notna(),
            relocated=lambda _df: _df.on_project & (_df.home_location != _df.location),
            employee_project_month=lambda _df: _df.employee_project_month.fillna(0),
            employee_percent_on_project=lambda _df: (_df.employee_project_month / _df.project_month).fillna(0),
            priority_project=lambda _df: _df.priority_project.astype("boolean").fillna(False).astype(bool),
        )
        .query("year.ge(@assets.MINIMUM_YEAR)")
    )


splitters = {
    keys.ML.TRAIN: splitter.get_train,
    keys.ML.TEST: splitter.get_test,
    keys.ML.VAL: splitter.get_val,
}

ml_data = ReadFromCache(
    storage.Dict(),
    ReadFromCache(
        piping.local_df_storage,
        piping.Builder(
            piping.local_df_storage,
            process,
            splitters,
        ),
    ),
)
