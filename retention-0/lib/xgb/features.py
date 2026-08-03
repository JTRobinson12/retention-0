from enum import StrEnum

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgb import ml_components


class EncoderNames(StrEnum):
    """Prevent bugs via StrEnum."""

    ONE_HOT = "one_hot"
    PASSTHROUGH = "passthrough"
    SCALER = "scaler"


encoders = {
    EncoderNames.ONE_HOT: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    EncoderNames.SCALER: StandardScaler(),
}

features = [
    ml_components.Feature(*args)
    for args in [
        ("tenure_years", EncoderNames.SCALER, "float64", "Tenure (Years)", True),
        ("age", EncoderNames.SCALER, "float64", "Age (Years)", True),
        ("male", EncoderNames.PASSTHROUGH, "bool", "Sex (Male)", True),
        ("ethnicity", EncoderNames.ONE_HOT, "category", "Ethnicity", True),
        ("rehire", EncoderNames.PASSTHROUGH, "bool", "Rehire", True),
        ("employee_percent_on_project", EncoderNames.SCALER, "float64", "Employee Percent on Project", True),
        ("relocated", EncoderNames.PASSTHROUGH, "bool", "Relocated", True),
        ("budget", EncoderNames.SCALER, "float64", "Project Size (Budget)", True),
        ("priority_project", EncoderNames.PASSTHROUGH, "bool", "Priority Project", True),
        ("on_project", EncoderNames.PASSTHROUGH, "bool", "On Project", True),
        ("relative_month", EncoderNames.SCALER, "float64", "Relative Month", True),
        ("employee_project_month", EncoderNames.SCALER, "float64", "Months on Project", True),
    ]
]

label_map = {f.name: f.label for f in features if f.label}
one_hot_list = [f.name for f in features if f.encoder == EncoderNames.ONE_HOT]
