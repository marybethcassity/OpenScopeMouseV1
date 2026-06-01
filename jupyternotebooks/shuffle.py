import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shuffle helpers
# ---------------------------------------------------------------------------

def shuffle_full(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Shuffle mode: 'full'
    Randomly permute every element of the response matrix independently,
    destroying all structure (unit identity AND condition structure).
    """
    flat = matrix.flatten()
    rng.shuffle(flat)
    return flat.reshape(matrix.shape)


def shuffle_sf_tf_within_neurons(matrix: np.ndarray, condition_ids: pd.Index,
                   stimulus_conditions_df_list: list,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Shuffle mode: 'sf_tf_full'
    For each unit (row), swap SF/TF tuning by permuting only the columns that
    correspond to different (sf, tf) pairs while not shuffling orientaiton
    Each neuron gets a different permutation. 

    """
    shuffled = matrix.copy()

    # Build a merged stimulus-conditions lookup keyed by condition_id
    merged_sc = pd.concat(stimulus_conditions_df_list)
    merged_sc = merged_sc[~merged_sc.index.duplicated(keep='first')]
    merged_sc = merged_sc.reindex(condition_ids)

    # Identify SF / TF column names flexibly
    sf_col = next((c for c in merged_sc.columns if c.lower() in ('sf', 'spatial_frequency')), None)
    tf_col = next((c for c in merged_sc.columns if c.lower() in ('tf', 'temporal_frequency')), None)
    ori_col = next((c for c in merged_sc.columns if c.lower() in ('ori', 'orientation',
                                                                    'direction', 'dir')), None)
    if sf_col is None or tf_col is None:
        # Fallback: shuffle every column independently per unit
        print("  [sf_tf shuffle] SF/TF columns not found – shuffling all columns per unit.")
        for i in range(shuffled.shape[0]):
            perm = rng.permutation(shuffled.shape[1])
            shuffled[i] = shuffled[i, perm]
        return shuffled

    if ori_col is not None:
        ori_values = merged_sc[ori_col].values
        unique_oris = np.unique(ori_values[~pd.isnull(ori_values)])
        for ori in unique_oris:
            col_idx = np.where(ori_values == ori)[0]
            if len(col_idx) < 2:
                continue
            # Each unit gets an independent permutation of the sf/tf columns
            # within this orientation
            for i in range(shuffled.shape[0]):
                perm = rng.permutation(len(col_idx))
                shuffled[i, col_idx] = shuffled[i, col_idx[perm]]
    else:
        # No orientation column – permute all sf/tf columns per unit
        print("  [sf_tf shuffle] No orientation column found – permuting all SF/TF cols per unit.")
        for i in range(shuffled.shape[0]):
            perm = rng.permutation(shuffled.shape[1])
            shuffled[i] = shuffled[i, perm]

    return shuffled

def shuffle_sf_tf_across_neurons(matrix: np.ndarray, condition_ids: pd.Index,
                   stimulus_conditions_df_list: list,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Shuffle mode: 'sf_tf_across_neurons'
    Shuffle SF/TF tuning across neurons by swapping which neurons get assigned to which SF/TF conditions, 
    while keeping the overall SF/TF structure intact. Keep ori with original neurons.
    """
    shuffled = matrix.copy()
    # Build a merged stimulus-conditions lookup keyed by condition_id
    merged_sc = pd.concat(stimulus_conditions_df_list)
    merged_sc = merged_sc[~merged_sc.index.duplicated(keep='first')]
    merged_sc = merged_sc.reindex(condition_ids)

    # Identify SF / TF column names flexibly
    sf_col = next((c for c in merged_sc.columns if c.lower() in ('sf', 'spatial_frequency')), None)
    tf_col = next((c for c in merged_sc.columns if c.lower() in ('tf', 'temporal_frequency')), None)
    ori_col = next((c for c in merged_sc.columns if c.lower() in ('ori', 'orientation',
                                                                    'direction', 'dir')), None)
    if sf_col is None or tf_col is None:
        print("  [sf_tf_across_neurons shuffle] SF/TF columns not found – returning matrix unchanged.")
        return matrix
    if ori_col is not None:
        print("  [sf_tf_across_neurons shuffle] Orientation column found – shuffling SF/TF across neurons while keeping ori structure.")
        # Group columns by orientation
        ori_values = merged_sc[ori_col].values
        unique_oris = np.unique(ori_values[~pd.isnull(ori_values)])
        for ori in unique_oris:
            col_idx = np.where(ori_values == ori)[0]
            if len(col_idx) < 2:
                continue
            # Shuffle which neurons get assigned to which SF/TF conditions within this orientation
            perm = rng.permutation(shuffled.shape[0])
            shuffled[:, col_idx] = shuffled[perm][:, col_idx]
    else:
        print("  [sf_tf_across_neurons shuffle] No orientation column found – shuffling SF/TF across neurons for all columns.")
        perm = rng.permutation(shuffled.shape[0])
        shuffled = shuffled[perm]

    return shuffled

def shuffle_sf_tf_ori_across_neurons(matrix: np.ndarray, condition_ids: pd.Index,
                   stimulus_conditions_df_list: list,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Shuffle mode: 'shuffle_sf_tf_ori_across_neurons'
    Shuffle SF/TF/ori tuning across neurons by swapping which neurons get assigned to which SF/TF/ori conditions, 
    while keeping the overall SF/TF/ori structure intact. 
    """
    shuffled = matrix.copy()
    
    # Build a merged stimulus-conditions lookup keyed by condition_id
    merged_sc = pd.concat(stimulus_conditions_df_list)
    merged_sc = merged_sc[~merged_sc.index.duplicated(keep='first')]
    merged_sc = merged_sc.reindex(condition_ids)
    
    # Identify SF / TF / ori column names flexibly
    sf_col = next((c for c in merged_sc.columns if c.lower() in ('sf', 'spatial_frequency')), None)
    tf_col = next((c for c in merged_sc.columns if c.lower() in ('tf', 'temporal_frequency')), None)
    ori_col = next((c for c in merged_sc.columns if c.lower() in ('ori', 'orientation',
                                                                    'direction', 'dir')), None)
    if sf_col is None or tf_col is None:
        print("  [shuffle_sf_tf_ori_across_neurons shuffle] SF/TF columns not found – returning matrix unchanged.")
        return matrix
    if ori_col is None:
        print("  [shuffle_sf_tf_ori_across_neurons shuffle] No orientation column found – shuffling SF/TF across neurons for all columns.")
        perm = rng.permutation(shuffled.shape[0])
        shuffled = shuffled[perm]
    else:
        print("  [shuffle_sf_tf_ori_across_neurons shuffle] Orientation column found – shuffling SF/TF/ori across neurons for all columns.")
        # Shuffle which neurons get assigned to which SF/TF/ori conditions
        perm = rng.permutation(shuffled.shape[0])
        shuffled = shuffled[perm]
    return shuffled

def shuffle_sf_tf_within_neurons_full(matrix, condition_ids, stimulus_conditions_df_list, rng):

    shuffled = matrix.copy()

    merged_sc = pd.concat(stimulus_conditions_df_list)
    merged_sc = merged_sc[~merged_sc.index.duplicated(keep='first')]
    merged_sc = merged_sc.reindex(condition_ids)

    sf_col = next((c for c in merged_sc.columns if c.lower() in ('sf', 'spatial_frequency')), None)
    tf_col = next((c for c in merged_sc.columns if c.lower() in ('tf', 'temporal_frequency')), None)

    if sf_col is None or tf_col is None:
        raise ValueError("SF/TF required")

    sf_values = merged_sc[sf_col].values

    unique_sf = np.unique(sf_values)
    sf_perm = rng.permutation(unique_sf)
    sf_map = dict(zip(unique_sf, sf_perm))

    perm_indices = np.empty(len(condition_ids), dtype=int)

    used = set()

    for old_sf in unique_sf:
        new_sf = sf_map[old_sf]

        src_idx = np.where(sf_values == old_sf)[0]
        tgt_idx = np.where(sf_values == new_sf)[0]

        # shuffle target indices ONCE
        tgt_idx = rng.permutation(tgt_idx)

        for j, i in enumerate(src_idx):
            perm_indices[i] = tgt_idx[j]

    for n in range(shuffled.shape[0]):
        shuffled[n] = shuffled[n, perm_indices]

    return shuffled


def drop_orientation(matrix: np.ndarray, condition_ids: pd.Index,
                               stimulus_conditions_df_list: list) -> tuple:
    """
    Shuffle mode: 'drop_orientation'
    Collapse orientation/direction by averaging responses across all orientations
    for each unique (sf, tf) combination (and any other non-orientation parameter).
    Returns (new_matrix, new_condition_ids) where new_condition_ids is a MultiIndex
    of the remaining stimulus parameters.

    Requires orientation / direction column in stimulus_conditions.
    If not found, returns the original matrix unchanged with a warning.
    """
    merged_sc = pd.concat(stimulus_conditions_df_list)
    merged_sc = merged_sc[~merged_sc.index.duplicated(keep='first')]
    merged_sc = merged_sc.reindex(condition_ids).reset_index()

    ori_col = next((c for c in merged_sc.columns
                    if c.lower() in ('ori', 'orientation', 'direction', 'dir')), None)

    if ori_col is None:
        print("  [drop_orientation] No orientation column found – returning matrix unchanged.")
        return matrix, condition_ids

    # Columns that define a condition after removing orientation
    drop_cols = [ori_col]
    id_col = merged_sc.index.name or 'index'   # the index column (condition_id)
    group_cols = [c for c in merged_sc.columns if c not in drop_cols + [id_col]]

    if not group_cols:
        print("  [drop_orientation] No remaining columns after dropping orientation – returning unchanged.")
        return matrix, condition_ids

    merged_sc['_orig_col_idx'] = np.arange(len(merged_sc))
    groups = merged_sc.groupby(group_cols, sort=True, observed=True)['_orig_col_idx'].apply(list)
    new_matrix = np.zeros((matrix.shape[0], len(groups)))
    new_condition_labels = []

    for j, (group_key, col_indices) in enumerate(groups.items()):
        new_matrix[:, j] = matrix[:, col_indices].mean(axis=1)
        new_condition_labels.append(group_key)

    new_condition_ids = pd.MultiIndex.from_tuples(
        new_condition_labels,
        names=group_cols
    )

    print(f"  [drop_orientation] Collapsed {matrix.shape[1]} → {new_matrix.shape[1]} conditions "
          f"by averaging over orientation.")
    return new_matrix, new_condition_ids

def drop_sf_tf(matrix: np.ndarray, condition_ids: pd.Index,
                               stimulus_conditions_df_list: list) -> tuple:
    """
    Shuffle mode: 'drop_sf_tf'
    Collapse SF/TF by averaging responses across all SF/TF combinations
    for each unique (orientation, direction) combination (and any other non-SF/TF parameter).
    Returns (new_matrix, new_condition_ids) where new_condition_ids is a MultiIndex
    of the remaining stimulus parameters.
    """
    merged_sc = pd.concat(stimulus_conditions_df_list)
    merged_sc = merged_sc[~merged_sc.index.duplicated(keep='first')]
    merged_sc = merged_sc.reindex(condition_ids).reset_index()

    sf_col = next((c for c in merged_sc.columns if c.lower() in ('sf', 'spatial_frequency')), None)
    tf_col = next((c for c in merged_sc.columns if c.lower() in ('tf', 'temporal_frequency')), None)    

    if sf_col is None or tf_col is None:
        print("  [drop_sf_tf] SF/TF columns not found – returning matrix unchanged.")
        return matrix, condition_ids

    drop_cols = [sf_col, tf_col]
    id_col = merged_sc.columns[0]   # the index column (condition_id)
    group_cols = [c for c in merged_sc.columns if c not in drop_cols + [id_col]]
    if not group_cols:
        print("  [drop_sf_tf] No remaining columns after dropping SF/TF – returning unchanged.")
        return matrix, condition_ids
    merged_sc['_orig_col_idx'] = np.arange(len(merged_sc))
    groups = merged_sc.groupby(group_cols, sort=False)['_orig_col_idx'].apply(list)
    new_matrix = np.zeros((matrix.shape[0], len(groups)))
    new_condition_labels = []
    for j, (group_key, col_indices) in enumerate(groups.items()):
        new_matrix[:, j] = matrix[:, col_indices].mean(axis=1)
        new_condition_labels.append(group_key)
    new_condition_ids = pd.Index(
        [str(k) for k in new_condition_labels],
        name='collapsed_condition'
    )
    print(f"  [drop_sf_tf] Collapsed {matrix.shape[1]} → {new_matrix.shape[1]} conditions "
          f"by averaging over SF/TF.")
    return new_matrix, new_condition_ids

