from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

FEATURES = ["hour_of_day", "amount", "ip_prefix", "login_frequency", "session_duration",
            "location_region", "purchase_pattern", "age_group"]
CLASSES = {"low_risk", "moderate_risk", "high_risk"}
NonnegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Category = Annotated[str, Field(min_length=1, max_length=80, pattern=r"\S")]


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour_of_day: int = Field(ge=0, le=23)
    amount: NonnegativeFloat = Field(description="Amount in dataset's simulated currency")
    ip_prefix: float = Field(ge=0, le=255, allow_inf_nan=False)
    login_frequency: int = Field(ge=0, le=1000000)
    session_duration: NonnegativeFloat = Field(description="Duration in minutes")
    location_region: Category
    purchase_pattern: Category
    age_group: Category


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    label: str
    class_scores: dict[str, float]
    training_mode: str
    model_version: str
    note: str = "Educational class scores; not calibrated fraud probabilities."


class BatchResponse(BaseModel):
    predictions: list[Prediction]
