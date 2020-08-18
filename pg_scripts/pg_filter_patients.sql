select 
    p.* 
from `physionet-data.mimic_core.patients` p 
    inner join `physionet-data.mimic_icu.icustays` i 
    on p.subject_id = i.subject_id 
where 
    i.los >= 3;
