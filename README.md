# OpenScopeMouseV1

Note: regression_combined.py has not been checked yet.

Documentation:
https://docs.google.com/document/d/1xQpvuW6So0hK6jcf-eHpWJDmTny2EWkjDUwV9LOwdTc/edit?usp=sharing

To Do: 
https://docs.google.com/document/d/1y0XpHaprXy6NmaaKtz32JZzTjowQHkf--x5kDP9GC0Q/edit?tab=t.0

## Jupyternotebooks --> ephys 

Run in this order:

`ephys_build_responsematrix.ipynb`:
generates `ephys_conditionwise_stats.csv` *(necessary for `ephys_compute_prefvariable`)*

--> `ephys_build_rf.ipynb`: (optional)
generates `ephys_rf_unit_info.csv`

--> `ephys_compute_prefvariable.ipynb`:
generates `ephys_pref_variables.csv`
