from enum import StrEnum


class Imported(StrEnum):
    """Container for keys of the raw simulated data sources."""

    PEOPLE = "people"
    PROJECTS = "projects"
    ASSIGNMENTS = "assignments"


class ML(StrEnum):
    """Container for keys of data used in our models."""

    TEST = "test"
    TRAIN = "train"
    VAL = "val"


class Scores(StrEnum):
    """Container for keys of scores from the models."""

    TERMINATED_SCORED = "terminated_scored"
    SHAPS = "shaps"
    TERMINATED_FEATURE_SHAPS = "terminated_feature_shaps"
