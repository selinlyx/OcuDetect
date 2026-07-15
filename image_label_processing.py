'''
ODIR-5K's LUT annotation file is patient level. This means image 
labels are based on both eyes of a single patient. So, an eye could 
be normal but the label could say cataract due to the other eye for 
the same patient having cataract. Thus, this script generates a
new LUT for the data that is labelled image level instead of patient
level. It also cleans out images that are labelled with labels
that indicate it is unusable (such as "low quality image").
'''

import re
import pandas as pd

INPUT_PATH = "ODIR-5K/data.xlsx"
OUTPUT_PATH = "ODIR-5K/image_labels.csv"

LABEL_COLUMNS = ["N", "D", "G", "C", "A", "H", "M", "O"]

# diagnostic keywords for each image -> disease label mapping 
KEYWORD_TO_LABELS = {
    "normal fundus": ["N"],
    "lens dust": ["N"],  # imaging artifact, not pathology
    "moderate non proliferative retinopathy": ["D"],
    "mild nonproliferative retinopathy": ["D"],
    "severe nonproliferative retinopathy": ["D"],
    "proliferative diabetic retinopathy": ["D"],
    "severe proliferative diabetic retinopathy": ["D"],
    "diabetic retinopathy": ["D"],
    "suspected diabetic retinopathy": ["D"],
    "suspicious diabetic retinopathy": ["D"],
    "suspected moderate non proliferative retinopathy": ["D"],
    "intraretinal microvascular abnormality": ["D"],
    "glaucoma": ["G"],
    "suspected glaucoma": ["G"],
    "cataract": ["C"],
    "suspected cataract": ["C"],
    "dry age-related macular degeneration": ["A"],
    "wet age-related macular degeneration": ["A"],
    "age-related macular degeneration": ["A"],
    "drusen": ["A"],
    "hypertensive retinopathy": ["H"],
    "pathological myopia": ["M"],
    "myopia retinopathy": ["M"],
    "myopic retinopathy": ["M"],
    "myopic maculopathy": ["M"],
    # everything below is bucketed under "Other" (O)
    "macular epiretinal membrane": ["O"],
    "epiretinal membrane": ["O"],
    "myelinated nerve fibers": ["O"],
    "laser spot": ["O"],
    "vitreous degeneration": ["O"],
    "refractive media opacity": ["O"],
    "spotted membranous change": ["O"],
    "tessellated fundus": ["O"],
    "chorioretinal atrophy": ["O"],
    "branch retinal vein occlusion": ["O"],
    "maculopathy": ["O"],
    "retinal pigmentation": ["O"],
    "white vessel": ["O"],
    "post retinal laser surgery": ["O"],
    "peripapillary atrophy": ["O"],
    "epiretinal membrane over the macula": ["O"],
    "retinitis pigmentosa": ["O"],
    "optic disc edema": ["O"],
    "central retinal vein occlusion": ["O"],
    "post laser photocoagulation": ["O"],
    "retinochoroidal coloboma": ["O"],
    "optic nerve atrophy": ["O"],
    "atrophic change": ["O"],
    "old branch retinal vein occlusion": ["O"],
    "depigmentation of the retinal pigment epithelium": ["O"],
    "chorioretinal atrophy with pigmentation proliferation": ["O"],
    "pigment epithelium proliferation": ["O"],
    "old chorioretinopathy": ["O"],
    "central retinal artery occlusion": ["O"],
    "retina fold": ["O"],
    "branch retinal artery occlusion": ["O"],
    "idiopathic choroidal neovascularization": ["O"],
    "old central retinal vein occlusion": ["O"],
    "abnormal pigment": ["O"],
    "rhegmatogenous retinal detachment": ["O"],
    "macular hole": ["O"],
    "atrophy": ["O"],
    "vessel tortuosity": ["O"],
    "punctate inner choroidopathy": ["O"],
    "intraretinal hemorrhage": ["O"],
    "fundus laser photocoagulation spots": ["O"],
    "morning glory syndrome": ["O"],
    "retinal pigment epithelial hypertrophy": ["O"],
    "pigmentation disorder": ["O"],
    "retinal pigment epithelium atrophy": ["O"],
    "suspected retinal vascular sheathing": ["O"],
    "macular coloboma": ["O"],
    "wedge white line change": ["O"],
    "old choroiditis": ["O"],
    "optic disk epiretinal membrane": ["O"],
    "asteroid hyalosis": ["O"],
    "diffuse chorioretinal atrophy": ["O"],
    "arteriosclerosis": ["O"],
    "silicone oil eye": ["O"],
    "choroidal nevus": ["O"],
    "diffuse retinal atrophy": ["O"],
    "macular pigmentation disorder": ["O"],
    "suspected microvascular anomalies": ["O"],
    "oval yellow-white atrophy": ["O"],
    "wedge-shaped change": ["O"],
    "congenital choroidal coloboma": ["O"],
    "retinal artery macroaneurysm": ["O"],
    "glial remnants anterior to the optic disc": ["O"],
    "vascular loops": ["O"],
    "optic discitis": ["O"],
    "retinal vascular sheathing": ["O"],
    "suspected retinitis pigmentosa": ["O"],
    "suspected abnormal color of optic disc": ["O"],
    "vitreous opacity": ["O"],
    "suspected macular epimacular membrane": ["O"],
    "retinal detachment": ["O"],
    "central serous chorioretinopathy": ["O"],
}

# keywords that indicate unusable / low quality images
QUALITY_FLAGS = {
    "low image quality",
    "no fundus image",
    "image offset",
    "anterior segment image",
    "optic disk photographically invisible",
}


def parse_keywords(raw_string):
    '''
    Splits all keyword phrases in a Diagnostic Keywords cell into individual keyword phases.
    Handles both '，' and plain ASCII comma, as well as spaces that show up in the 
    original data csv file. 

    '''

    if pd.isna(raw_string):
        return []
    parts = re.split("[，,]", str(raw_string))
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]


def labels_for_eye(raw_keywords):
    '''
    Return (label_dict, is_droppable) for one eye's keyword string.
    
    '''
    tokens = parse_keywords(raw_keywords)
    label_vec = {col: 0 for col in LABEL_COLUMNS}

    for token in tokens:
        if token in QUALITY_FLAGS:
            return label_vec, True  # drop: not a usable fundus image
        if token not in KEYWORD_TO_LABELS:
            print(f"  [warning] unrecognized keyword -> defaulting to 'O': '{token}'")
            label_vec["O"] = 1
            continue
        for label in KEYWORD_TO_LABELS[token]:
            label_vec[label] = 1

    return label_vec, False


def build_image_label_table(df):
    records = []
    dropped = []

    for _, row in df.iterrows():
        for side, fname_col, kw_col in [
            ("left", "Left-Fundus", "Left-Diagnostic Keywords"),
            ("right", "Right-Fundus", "Right-Diagnostic Keywords"),
        ]:
            filename = row[fname_col]
            label_vec, drop = labels_for_eye(row[kw_col])
            if drop:
                dropped.append(filename)
                continue
            # carry patient-level metadata through so downstream code can
            # split/stratify by patient (or by sex/age) without having to
            # re-parse patient id back out of the filename.
            record = {
                "filename": filename,
                "patient_id": row["ID"],
                "age": row["Patient Age"],
                "sex": row["Patient Sex"],
                "eye": side,
            }
            record.update(label_vec)
            records.append(record)

    print(f"\nDropped {len(dropped)} images due to quality/artifact flags:")
    for f in dropped:
        print(f"  - {f}")

    return pd.DataFrame(records)


def main():
    df = pd.read_excel(INPUT_PATH)
    print(f"Loaded {len(df)} patient rows from {INPUT_PATH}")

    image_labels = build_image_label_table(df)
    image_labels.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(image_labels)} image-level records to {OUTPUT_PATH}")
    print("\nLabel prevalence (per image):")
    print(image_labels[LABEL_COLUMNS].sum())
    label_sum = image_labels[LABEL_COLUMNS].sum(axis=1)
    print("\nLabels per image (multi-label check):")
    print(label_sum.value_counts().sort_index())


if __name__ == "__main__":
    main()