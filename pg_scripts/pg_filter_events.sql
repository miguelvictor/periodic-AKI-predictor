select
    labevents.subject_id,
    labevents.hadm_id,
    icustays.icustay_id,
    labevents.itemid,
    labevents.charttime,
    labevents.value,
    labevents.valuenum,
    labevents.valueuom
from labevents inner join icustays on 
    labevents.subject_id = icustays.subject_id
    and labevents.hadm_id = icustays.hadm_id
    and labevents.charttime >= icustays.intime
    and labevents.charttime <= icustays.outtime
where itemid in (
        50882, -- bicarbonate (mEq/L)
        50902, -- chloride
        50912, -- creatinine
        50931, -- glucose
        50960, -- magnesium
        50822, 50971, -- potassium (mEq/L)
        50824, 50983, -- sodium (mEq/L == mmol/L)
        51006, -- blood urea nitrogen
        51222, -- hemoglobin
        51265, -- platelets
        51300, 51301, -- white blood cells
        51081 -- serum creatinine
    )
    and icustays.los >= 3

union all

select 
    chartevents.subject_id,
    chartevents.hadm_id,
    chartevents.icustay_id,
    chartevents.itemid,
    chartevents.charttime,
    chartevents.value,
    chartevents.valuenum,
    chartevents.valueuom
from chartevents inner join icustays on 
    chartevents.icustay_id = icustays.icustay_id
where itemid IN (
        226707, -- height (inches)
        226730, -- height (cm)
        1394, -- height (inches)
        763, -- weight (kg)
        224639 -- weight (kg)
    )
    and icustays.los >= 3