select 
    a.* 
from `physionet-data.mimic_core.admissions` a 
    inner join `physionet-data.mimic_icu.icustays` i 
    on a.hadm_id = i.hadm_id 
where 
    i.los >= 3;
