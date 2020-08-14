select
    l.subject_id,
    l.hadm_id,
    i.icustay_id,
    l.itemid,
    l.charttime,
    l.valuenum,
    l.valueuom
from labevents l inner join icustays i on 
    l.subject_id = i.subject_id
    and l.hadm_id = i.hadm_id
    and l.charttime >= i.intime
    and l.charttime <= i.outtime
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
        51300, 51301 -- white blood cells
    )
    and i.los >= 3
    and l.valuenum is not null

union all

select 
    c.subject_id,
    c.hadm_id,
    c.icustay_id,
    c.itemid,
    c.charttime,
    c.valuenum,
    c.valueuom
from chartevents c inner join icustays i on 
    c.icustay_id = i.icustay_id
where itemid IN (
        226707, -- height (inches)
        226730, -- height (cm)
        1394, -- height (inches)
        763, -- weight (kg)
        224639 -- weight (kg)
    )
    and i.los >= 3
    and c.valuenum is not null
