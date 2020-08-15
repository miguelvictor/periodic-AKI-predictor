(
    select distinct on (i.icustay_id)
        l.subject_id,
        l.hadm_id,
        i.icustay_id,
        l.charttime,
        l.valuenum,
        l.valueuom,
        'first' as timing
    from labevents l inner join icustays i on
        l.subject_id = i.subject_id
        and l.hadm_id = i.hadm_id
        and l.charttime >= i.intime
        and l.charttime <= i.outtime
    where itemid = 50912
        and i.los >= 3
        and l.valuenum is not null
    order by i.icustay_id, charttime asc
)

union all

(
    select distinct on (i.icustay_id)
        l.subject_id,
        l.hadm_id,
        i.icustay_id,
        l.charttime,
        l.valuenum,
        l.valueuom,
        'last' as timing
    from labevents l inner join icustays i on
        l.subject_id = i.subject_id
        and l.hadm_id = i.hadm_id
        and l.charttime >= i.intime
        and l.charttime <= i.outtime
    where itemid = 50912
        and i.los >= 3
        and l.valuenum is not null
    order by i.icustay_id, charttime desc
)