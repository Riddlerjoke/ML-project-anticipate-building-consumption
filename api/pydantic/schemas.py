# api/pydantic/schemas.py
from pydantic import BaseModel, Field, constr, confloat, conint


class BuildingFeatures(BaseModel):
    DataYear: conint(ge=2000, le=2100)
    BuildingType: constr(strip_whitespace=True, min_length=1)
    PrimaryPropertyType: constr(strip_whitespace=True, min_length=1)
    Neighborhood: constr(strip_whitespace=True, min_length=1)
    NumberofBuildings: confloat(gt=0)
    NumberofFloors: confloat(gt=0)
    PropertyGFATotal: confloat(gt=0)
    PropertyGFAParking: confloat(ge=0)
    PropertyGFABuilding_s: confloat(gt=0) = Field(alias="PropertyGFABuilding(s)")
    LargestPropertyUseTypeGFA: confloat(gt=0)
    BuildingAge: conint(ge=0, le=300)

    class Config:
        validate_by_name = True
