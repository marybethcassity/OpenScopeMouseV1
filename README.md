# OpenScopeMouseV1

Note: regression_combined.py has not been checked yet.

Documentation:
https://docs.google.com/document/d/1xQpvuW6So0hK6jcf-eHpWJDmTny2EWkjDUwV9LOwdTc/edit?usp=sharing

To Do: 
https://docs.google.com/document/d/1y0XpHaprXy6NmaaKtz32JZzTjowQHkf--x5kDP9GC0Q/edit?tab=t.0


### Run the code from a command line interface (e.g. Anaconda)
1. First, activate your anaconda environment  
    `conda activate your_environment_name`
2. Navigate to the directory containing compute_pref_variables.py and run:  
    `python compute_pref_variables.py --data_dir path/to/data/folder --output_dir path/to/output/folder`

    - Ex. `python compute_pref_variables.py --data_dir "X:\Personnel\MaryBeth\OpenScope\001568" --filtered --output_dir "X:\Personnel\MaryBeth\OpenScope\001568\results"`

3. Optional arguments can be found in `compute_pref_variables.py`

### Workflow
1. Run the `main()` method in compute_pref_variables.py
    - Parses arguments
    - Gets list of subdirectories that start with ‘sub-’
    - Creates output directory in results/ with arguments in the name 
    - Initializes storage for all subdirectories 
    - Processes each subdirectory in the list 
    - Creates output directory 
    - Finds the NWB file 
    - Loads the NWB data (nwb, units, receptive field stimulus table, & drifting gratings stimulus table) and saves in dictionary nwb_data
    - Initializes storage for all probes to analyse
    - Creates output directory for each probe to analyze
    - Calls process_units() from data_processing.py to get the units for the probe
2. data_processing.py → process_units()
    - Gets receptive field x and y positions
    - Filters the units by probe 
    - Calls get_rf() from data_processing.py
3. data_processing.py → get_rf()
    - Calculates the receptive field from the spike times from the receptive field stimulus table
4. data_processing.py → process_units()
    - Returns dictionary units_data with unit_indices, unit_rfs, xs, and ys
5. compute_pref_variables.py → main()
    - If flag - -filtered is passed, calls filter_rfs_by_gaussian_fit() from gaussian_filtering.py
6. gaussian_filtering.py → filter_rfs_by_gaussian_fit()
    - Fill in notes here 
7. compute_pref_variables.py → main()
    - Updates dictionary units_data with unit_rfs:filtered_rfs and unit_indices:filtered_indices 
    - Calls calculate_all_metrics() from data_processing.py
8. data_processing.py → calculate_all_metrics()
    - For each unit units_data[‘unit_indices’] calls calculate_preferred_metrics() from data_processing.py
9. data_processing.py → calculate_preferred_metrics()
