# there are two splitting goals 
# 1. no patient appears in two splits > the point of this 
#    is to ensure that the model does not learn patients
# 2. have the same proportion of each class in each split >
#    the point of this is to ensure we don't randomly 
#    split too much or nearly all of a class into a single split

import numpy as np
import pandas as pd
from skmultilearn.model_selection import IterativeStratification

INPUT_CSV = "ODIR-5K/image_labels.csv"
TRAIN_CSV = "ODIR-5K/train_labels.csv"
VAL_CSV = "ODIR-5K/val_labels.csv"
TEST_CSV = "ODIR-5K/test_labels.csv"

LABEL_COLUMNS = ["N", "D", "G", "C", "A", "H", "M", "O"]
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42

def aggregate_patient_labels(df):
    '''
    Aggregates data such that each row contains the combined diagnoses 
    for one patient, rather than per image. This is used to decide the 
    split membership.
    '''
    # this groups the data by the patient id 
    return df.groupby("patient_id")[LABEL_COLUMNS].max().reset_index()

# because the data is multi label, we cannot use simple single stratification
# thus, we use iterative stratification 

def iterative_split(patient_labels, fractions):
    '''
    Splits patient labels into train, validate, and test sets
    based on the fractions for each set using iterative stratification.

    patient_labels -> df of aggregated disease labels by patient
    fractions -> [fraction train, fraction validation, fraction test]

    '''
    y = patient_labels[LABEL_COLUMNS].values # get the values of each row
    ids = patient_labels["patient_id"].values # get the patient IDs

    remaining_ids, remaining_y = ids, y
    groups = [] # contains the test, validation, test splits 
    remaining_fraction = 1.0 # remaining to be split

    for frac in fractions[:-1]:
        split_ratio = frac / remaining_fraction  # this group's share of what's left
        stratifier = IterativeStratification(n_splits=2, order=2,
            sample_distribution_per_fold=[split_ratio, 1 - split_ratio],)
        index_leftover, index_split = next(stratifier.split(remaining_ids.reshape(-1, 1), remaining_y))
        groups.append(remaining_ids[index_split]) # add to groups the split off amount
        remaining_ids, remaining_y = remaining_ids[index_leftover], remaining_y[index_leftover] # update remaining data
        remaining_fraction -= frac 

    groups.append(remaining_ids)  # whatever's left is the last group
    return groups


def perform_splitting():
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} images across {df['patient_id'].nunique()} patients.") # check number of images and unqiue patients

    patient_labels = aggregate_patient_labels(df) # aggregate data by patients, so when we split no same patient shows up in two sets
    train_ids, val_ids, test_ids = iterative_split(patient_labels, [TRAIN_FRAC, VAL_FRAC, TEST_FRAC]) # perform iterative stratification

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    out_paths = {"train": TRAIN_CSV, "val": VAL_CSV, "test": TEST_CSV}

    for set, ids in splits.items():
        split_df = df[df["patient_id"].isin(ids)] # get the data for patient ids that are in the current set's ids
        # note we are using the original df, so now image level and not patient level,
        # but we are still using patient id to decide membership in splits to ensure no patient overlap

        split_df.to_csv(out_paths[set], index=False) # turn into a csv file
        # check how many patients and images went into each split set
        print(f"\n{set}: {len(ids)} patients, {len(split_df)} images -> {out_paths[set]}")

    # compare label prevalence as % of that split's images
    print("\nLabel prevalence by split (% of images in that split):")
    summary = pd.DataFrame({
        set: df[df["patient_id"].isin(ids)][LABEL_COLUMNS].mean() * 100
        for set, ids in splits.items()
    })
    print(summary.round(2))

    # confirm no patient appears in more than one split
    all_assigned = set(train_ids) | set(val_ids) | set(test_ids) # union of unique ids from sets
    # if the total count of ids in each set equal to the count of ids in the union of unique sets, then no overlap
    no_overlap = len(train_ids) + len(val_ids) + len(test_ids) == len(all_assigned) 
    print(f"\nNo patient overlap across splits: {no_overlap}")


if __name__ == "__main__":
    perform_splitting()