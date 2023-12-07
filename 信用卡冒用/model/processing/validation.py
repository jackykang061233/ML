from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Check model inputs for na values and filter.

    Parameters:
    - input_data(pd.DataFrame): the input DataFrame

    Returns:
    - pd.DataFrame: the validated data
    """
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in input_data.columns
        if var
        not in config.log_config.vars_with_na
        + config.log_config.add_na_column
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Check model inputs for unprocessable values

    Parameters:
    - input_data(pd.DataFrame): the input DataFrame

    Returns:
    - Tuple[pd.DataFrame, Optional[dict]]: the validated data   
    """

    features = [col for col in input_data.columns if col not in config.log_config.to_drop]
    relevant_data = input_data[features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        Inputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error: 
        errors = error.json()

    return validated_data, errors


class InputSchema(BaseModel):
    """ The format of input data """
    txkey: Optional[str]
    locdt: Optional[int]
    loctm: Optional[int]
    chid: Optional[str]
    cano: Optional[str]
    contp: Optional[int]
    etymd: Optional[float]
    mchno: Optional[str]
    acqic: Optional[str]
    mcc: Optional[float]
    conam: Optional[float]
    ecfg: Optional[int]
    insfg: Optional[int]
    iterm: Optional[float]
    bnsfg: Optional[int]
    flam1: Optional[int]
    stocn: Optional[float]
    scity: Optional[float]
    stscd: Optional[float]
    ovrlt: Optional[int]
    flbmk: Optional[int]
    hcefg: Optional[float]
    csmcu: Optional[float]
    csmam: Optional[int]
    flg_3dsmk: Optional[int]
    label: Optional[int]

class Inputs(BaseModel):
    inputs: List[InputSchema]