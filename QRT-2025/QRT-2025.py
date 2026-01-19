# ===================== QRT — Twin V2.22 (v2.16 + imputation mol 0 + KNN clinique + KNN global) =====================
import os, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis

import xgboost as xgb
import lightgbm as lgb

RNG = 42
N_FOLDS = 5
CV_SEEDS = [42, 141, 314]

# ----------------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------------
def rp(path):
    for base in ("../data", "./", "/mnt/data"):
        p = os.path.join(base, path)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(path)

def to_rank(x):
    s = pd.Series(x, index=range(len(x)))
    r = s.rank(method="average").values
    return (r - r.min()) / (r.max() - r.min() + 1e-12)

def winsorize(s, p=0.01):
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def gini_coeff(a):
    a = np.asarray([v for v in a if not pd.isna(v)], dtype=float)
    if a.size == 0:
        return np.nan
    if np.allclose(a, 0):
        return 0.0
    a = np.sort(a)
    n = a.size
    cum = np.cumsum(a)
    return (n + 1 - 2 * (cum / cum[-1]).sum()) / n

def normalize_effect(s):
    s = str(s).lower()
    if "frame" in s: return "frameshift"
    if "stop" in s: return "nonsense"
    if "splice" in s: return "splice"
    if "synonymous" in s: return "synonymous"
    if "missense" in s or "non_synonymous" in s or "non-synonymous" in s:
        return "missense_like"
    return "other"

def vaf_slopes_series(s):
    arr = np.sort(s.dropna().values)[::-1]  # descending
    if arr.size == 0:
        return pd.Series(
            {"VAF_slope_12": np.nan, "VAF_slope_23": np.nan, "VAF_slope_13": np.nan}
        )
    v1 = arr[0]
    v2 = arr[1] if arr.size > 1 else arr[-1]
    v3 = arr[2] if arr.size > 2 else arr[-1]
    return pd.Series(
        {
            "VAF_slope_12": float(v1 - v2),
            "VAF_slope_23": float(v2 - v3),
            "VAF_slope_13": float(v1 - v3),
        }
    )

# ----------------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------------
clin_tr = pd.read_csv(rp("clinical_train.csv"))
clin_te = pd.read_csv(rp("clinical_test.csv"))
mol_tr  = pd.read_csv(rp("molecular_train.csv"))
mol_te  = pd.read_csv(rp("molecular_test.csv"))
y_tr    = pd.read_csv(rp("target_train.csv"))

print("=== SHAPES RAW ===")
print(f"clinical_train: {clin_tr.shape}  clinical_test: {clin_te.shape}")
print(f"molecular_train: {mol_tr.shape}  molecular_test: {mol_te.shape}")
print(f"target_train: {y_tr.shape}")

# ----------------------------------------------------------------------
# TARGETS (truncate 7y)
# ----------------------------------------------------------------------
y = y_tr.copy()
y["time"]  = np.minimum(y["OS_YEARS"], 7.0)
y["event"] = ((y["OS_STATUS"] == 1) & (y["OS_YEARS"] <= 7.0)).astype(int)
y = y[y["time"].notna()][["ID", "time", "event"]].set_index("ID")

y_sks = Surv.from_arrays(
    event=y["event"].astype(bool).values,
    time=y["time"].astype(float).values,
)

# ----------------------------------------------------------------------
# CYTO FEATURES
# ----------------------------------------------------------------------
CYTO_FLAGS = {
    "monosomy7": r"(^|[,;\s])-7(?!\d)",
    "del7q":     r"del\(7q\)|7q-",
    "monosomy5": r"(^|[,;\s])-5(?!\d)",
    "del5q":     r"del\(5q\)|5q-",
    "t_8_21":    r"t\(8;21\)",
    "inv16_t16": r"inv\(16\)|t\(16;16\)",
    "t_15_17":   r"t\(15;17\)",
    "trisomy8":  r"(^|[,;\s])\+8(?!\d)",
    "inv3_t3_3": r"inv\(3\)|t\(3;3\)",
    "+13":       r"(^|[,;\s])\+13(?!\d)",
    "del20q":    r"del\(20\)|20q-",
}

def parse_cyto(s: str):
    t = str(s).lower() if pd.notnull(s) else ""
    feats = {k: int(bool(re.search(p, t))) for k, p in CYTO_FLAGS.items()}
    normal = bool(re.search(r"46,xx|46,xy", t))
    token_count = (len(re.findall(r"[,;/]", t)) + 1) if t else 0
    feats["normal_karyotype"]  = int(normal and not any(feats.values()))
    feats["cyto_token_count"]  = token_count
    feats["complex_karyotype"] = int(token_count >= 3)
    feats["sex_XY"] = int("46,xy" in t)
    feats["sex_XX"] = int("46,xx" in t)
    return pd.Series(feats)

for df in (clin_tr, clin_te):
    cy = df["CYTOGENETICS"].apply(parse_cyto)
    for c in cy.columns:
        df[c] = cy[c]

# ----------------------------------------------------------------------
# CLINICAL TRANSFORMS + MISSINGNESS
# ----------------------------------------------------------------------
CLIN_CONT = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]

for df in (clin_tr, clin_te):
    # missingness flags
    for c in CLIN_CONT:
        df[f"MISS_{c}"] = df[c].isna().astype(int)

    # log-transform certains labos
    for c in ["WBC", "ANC", "MONOCYTES", "PLT", "BM_BLAST"]:
        df[c] = np.log1p(df[c])

    # winsorisation
    for c in CLIN_CONT:
        df[c] = winsorize(df[c])

# ----------------------------------------------------------------------
# MOLECULAR PREPROCESSING
# ----------------------------------------------------------------------
for df in (mol_tr, mol_te):
    df["EFFECT_N"] = df["EFFECT"].map(normalize_effect)
    df["VAF"]      = pd.to_numeric(df["VAF"], errors="coerce").clip(0, 1)
    df["DEPTH"]    = pd.to_numeric(df.get("DEPTH", np.nan), errors="coerce")
    df["GENE"]     = df["GENE"].astype(str)

# gènes fréquents (train+test)
gene_freq_all = pd.concat([mol_tr["GENE"], mol_te["GENE"]]).value_counts()
TOP_N_GENES   = 70
top_genes = set(gene_freq_all.head(TOP_N_GENES).index)

# panel AML curated & pathways
CURATED = [
    "NPM1","FLT3","DNMT3A","TET2","ASXL1","RUNX1",
    "IDH1","IDH2","TP53","SRSF2","SF3B1","U2AF1",
    "NRAS","KRAS","CBL","PTPN11","KIT","CEBPA"
]

# myelodysplasia-related genes (ELN adverse)
MR_GENES = ["ASXL1","BCOR","EZH2","RUNX1","SF3B1","SRSF2","STAG2","U2AF1","ZRSR2"]
CURATED = sorted(set(CURATED + MR_GENES))
top_genes |= set(CURATED)

PATH_RAS = {"NRAS", "KRAS", "PTPN11", "CBL"}
PATH_EPI = {"DNMT3A", "TET2", "ASXL1", "IDH1", "IDH2"}
PATH_SPL = {"SRSF2", "SF3B1", "U2AF1"}

def aggregate_mol(df):
    g = df.groupby("ID")

    # base aggregates
    base = g.agg(
        n_vars=("GENE", "count"),
        n_genes=("GENE", "nunique"),
        VAF_sum=("VAF", "sum"),
        VAF_max=("VAF", "max"),
        VAF_mean=("VAF", "mean"),
        VAF_std=("VAF", "std"),
        DEPTH_mean=("DEPTH", "mean"),
        DEPTH_median=("DEPTH", "median"),
    )

    # VAF shape
    vaf_gini = g["VAF"].apply(gini_coeff).rename("VAF_gini")
    vaf_top3_mean = g["VAF"].apply(
        lambda s: np.mean(sorted(s.dropna())[-3:]) if s.notna().any() else np.nan
    ).rename("VAF_top3_mean")
    prop_vaf_lt_005 = g["VAF"].apply(
        lambda s: np.mean(s.dropna() < 0.05) if s.notna().any() else np.nan
    ).rename("prop_VAF_lt_005")
    prop_vaf_ge_035 = g["VAF"].apply(
        lambda s: np.mean(s.dropna() >= 0.35) if s.notna().any() else np.nan
    ).rename("prop_VAF_ge_035")

    # slopes
    slopes = g["VAF"].apply(vaf_slopes_series)

    # thresholds
    n_ge_035 = g.apply(lambda x: (x["VAF"] >= 0.35).sum()).rename("n_mut_vaf_ge_035")
    n_ge_045 = g.apply(lambda x: (x["VAF"] >= 0.45).sum()).rename("n_mut_vaf_ge_045")

    # effect counts
    eff = df.pivot_table(
        index="ID", columns="EFFECT_N", values="VAF", aggfunc="count"
    ).fillna(0.0)
    eff.columns = [f"eff_{c}" for c in eff.columns]

    # curated presence (inclut MR_GENES)
    pres = df[df["GENE"].isin(CURATED)].pivot_table(
        index="ID", columns="GENE", values="VAF", aggfunc="max"
    )
    if pres.size > 0:
        pres = (pres.notna()).astype(int)
        pres.columns = [f"has__{c}" for c in pres.columns]
    else:
        pres = pd.DataFrame(index=base.index)

    # pathways
    path_ras = g.apply(lambda x: x["GENE"].isin(PATH_RAS).sum()).rename("path_RAS_count")
    path_epi = g.apply(lambda x: x["GENE"].isin(PATH_EPI).sum()).rename("path_EPI_count")
    path_spl = g.apply(lambda x: x["GENE"].isin(PATH_SPL).sum()).rename("path_SPL_count")

    # per-gene VAFmax (top_genes)
    df_top = df[df["GENE"].isin(top_genes)]
    by_gene = df_top.groupby(["ID", "GENE"]).agg(VAF_max=("VAF", "max")).reset_index()
    if not by_gene.empty:
        vaf_max = by_gene.pivot(index="ID", columns="GENE", values="VAF_max")
        vaf_max.columns = [f"VAFmax__{g}" for g in vaf_max.columns]
    else:
        vaf_max = pd.DataFrame(index=base.index)

    # assemble all
    agg = base.join(
        [
            vaf_gini,
            vaf_top3_mean,
            prop_vaf_lt_005,
            prop_vaf_ge_035,
            slopes,
            n_ge_035,
            n_ge_045,
            eff,
            pres,
            path_ras,
            path_epi,
            path_spl,
            vaf_max,
        ],
        how="left",
    )

    # clonal_potential & baseline genomic_risk_score
    agg["clonal_potential"] = agg["n_vars"] * agg["VAF_sum"]

    tp53 = agg.get("has__TP53", 0)
    ras  = agg.get("path_RAS_count", 0)
    epi  = agg.get("path_EPI_count", 0)
    agg["genomic_risk_score"] = 2 * tp53 + ras + epi

    # MR gene burden
    mr_cols = [c for c in agg.columns if c.startswith("has__") and c[5:] in MR_GENES]
    if mr_cols:
        agg["mr_gene_mut_count"] = agg[mr_cols].sum(axis=1)
        agg["mr_gene_mut_any"]   = (agg["mr_gene_mut_count"] > 0).astype(int)
    else:
        agg["mr_gene_mut_count"] = 0
        agg["mr_gene_mut_any"]   = 0

    # NPM1-like favorable profile (approx ELN)
    has_npm1 = agg["has__NPM1"] if "has__NPM1" in agg.columns else 0
    has_flt3 = agg["has__FLT3"] if "has__FLT3" in agg.columns else 0
    has_tp53 = agg["has__TP53"] if "has__TP53" in agg.columns else 0
    mr_any   = agg["mr_gene_mut_any"]

    npm1_like_fav = (
        (has_npm1 == 1)
        & (has_flt3 == 0)
        & (has_tp53 == 0)
        & (mr_any == 0)
    )
    agg["npm1_like_favorable"] = npm1_like_fav.astype(int)

    # adverse_mol flag
    agg["adverse_mol_flag"] = ((has_tp53 == 1) | (mr_any == 1)).astype(int)

    return agg

mol_tr_agg = aggregate_mol(mol_tr)
mol_te_agg = aggregate_mol(mol_te)

# ----------------------------------------------------------------------
# BUILD X + ELN-LIKE RISK
# ----------------------------------------------------------------------
def build_matrix(clin, mol_agg):
    X = clin.set_index("ID").join(mol_agg, how="left")

    adv_cyto_cols = [
        "monosomy7", "del7q", "monosomy5", "del5q",
        "inv3_t3_3", "del20q", "complex_karyotype"
    ]
    fav_cyto_cols = ["t_8_21", "inv16_t16"]

    present_adv = [c for c in adv_cyto_cols if c in X.columns]
    present_fav = [c for c in fav_cyto_cols if c in X.columns]

    if present_adv:
        adv_cyto = (X[present_adv].sum(axis=1) > 0).astype(int)
    else:
        adv_cyto = pd.Series(0, index=X.index)

    if present_fav:
        fav_cyto = (X[present_fav].sum(axis=1) > 0).astype(int)
    else:
        fav_cyto = pd.Series(0, index=X.index)

    adv_mol  = X["adverse_mol_flag"] if "adverse_mol_flag" in X.columns else 0
    fav_npm1 = X["npm1_like_favorable"] if "npm1_like_favorable" in X.columns else 0

    eln_adv = ((adv_cyto == 1) | (adv_mol == 1)).astype(int)
    eln_fav = ((fav_cyto == 1) | (fav_npm1 == 1)).astype(int)
    eln_int = ((eln_adv == 0) & (eln_fav == 0)).astype(int)

    X["eln_like_favorable"]    = eln_fav
    X["eln_like_intermediate"] = eln_int
    X["eln_like_adverse"]      = eln_adv

    return X

X_tr_raw = build_matrix(clin_tr, mol_tr_agg)
X_te_raw = build_matrix(clin_te, mol_te_agg)

# align IDs with target
train_ids = X_tr_raw.index.intersection(y.index)
X_tr_raw = X_tr_raw.loc[train_ids].copy()
y_sub = y.loc[train_ids]
y_sks = Surv.from_arrays(
    event=y_sub["event"].astype(bool).values,
    time=y_sub["time"].astype(float).values,
)

# align columns train/test (before CENTER one-hot)
all_cols = sorted(set(X_tr_raw.columns) | set(X_te_raw.columns))
X_tr_raw = X_tr_raw.reindex(columns=all_cols)
X_te_raw = X_te_raw.reindex(columns=all_cols)

# ----------------------------------------------------------------------
# Domain-aware imputation pour les agrégats moléculaires
#   - VAFmax__GENE: NaN = pas de mutation => 0.0
#   - Burdens / scores moléculaires: NaN => 0.0
# ----------------------------------------------------------------------
vafmax_cols = [c for c in X_tr_raw.columns if c.startswith("VAFmax__")]

mol_zero_candidates = [
    "n_vars", "n_genes",
    "VAF_sum",
    "clonal_potential",
    "genomic_risk_score",
    "mr_gene_mut_count",
    "mr_gene_mut_any",
    "path_RAS_count",
    "path_EPI_count",
    "path_SPL_count",
    "npm1_like_favorable",
    "adverse_mol_flag",
]
mol_zero_cols = [c for c in mol_zero_candidates if c in X_tr_raw.columns]

for df in (X_tr_raw, X_te_raw):
    if vafmax_cols:
        df[vafmax_cols] = df[vafmax_cols].fillna(0.0)
    if mol_zero_cols:
        df[mol_zero_cols] = df[mol_zero_cols].fillna(0.0)

# ----------------------------------------------------------------------
# KNN IMPUTER ciblé sur les variables cliniques continues
#   (après logs & winsorisation, avant one-hot CENTER & imputation globale)
# ----------------------------------------------------------------------
clin_knn_cols = [c for c in CLIN_CONT if c in X_tr_raw.columns]

if clin_knn_cols:
    knn_imp = KNNImputer(
        n_neighbors=5,
        weights="distance",
        metric="nan_euclidean",
    )
    X_tr_raw[clin_knn_cols] = knn_imp.fit_transform(X_tr_raw[clin_knn_cols])
    X_te_raw[clin_knn_cols] = knn_imp.transform(X_te_raw[clin_knn_cols])

# ----------------------------------------------------------------------
# CENTER one-hot encoding
# ----------------------------------------------------------------------
if "CENTER" in X_tr_raw.columns:
    centers_all = pd.concat([X_tr_raw["CENTER"], X_te_raw["CENTER"]]).astype(str)
    center_dummies = pd.get_dummies(centers_all, prefix="CENTER")
    center_tr = center_dummies.loc[X_tr_raw.index]
    center_te = center_dummies.loc[X_te_raw.index]
    X_tr_raw = X_tr_raw.join(center_tr)
    X_te_raw = X_te_raw.join(center_te)
    X_tr_raw.drop(columns=["CENTER"], inplace=True)
    X_te_raw.drop(columns=["CENTER"], inplace=True)

# drop all-NaN & almost-empty
na_all_train = X_tr_raw.columns[X_tr_raw.isna().all()]
X_tr_raw.drop(columns=na_all_train, inplace=True)
X_te_raw.drop(columns=na_all_train, inplace=True)

few_data_cols = X_tr_raw.columns[(X_tr_raw.notna().sum() < 5)]
X_tr_raw.drop(columns=few_data_cols, inplace=True)
X_te_raw.drop(columns=few_data_cols, inplace=True)

# drop raw text cyto only
for df in (X_tr_raw, X_te_raw):
    if "CYTOGENETICS" in df.columns:
        df.drop(columns="CYTOGENETICS", inplace=True)

print(f"Features before clean: {len(all_cols)}")
print(f"Features final (before scaling): {X_tr_raw.shape[1]}")

# ----------------------------------------------------------------------
# IMPUTE (KNN global au max) + SCALE + VARIANCE FILTER + DROP HIGH CORR
# ----------------------------------------------------------------------
# 1) KNN global sur les features continues non binaires (hors colonnes protégées)
num_cols = X_tr_raw.select_dtypes(include=[np.number]).columns.tolist()

# colonnes binaires 0/1 à ne PAS passer au KNN global
binary_cols = []
for c in num_cols:
    vals = pd.unique(X_tr_raw[c].dropna())
    if len(vals) > 0 and len(vals) <= 2 and set(vals).issubset({0, 1, 0.0, 1.0}):
        binary_cols.append(c)
binary_cols = set(binary_cols)

# colonnes protégées où NaN=0 a un sens métier (molaires)
protected_zero_cols = set(vafmax_cols) | set(mol_zero_cols)

clin_knn_set = set(clin_knn_cols)

cont_knn_cols = []
for c in num_cols:
    if c in binary_cols:
        continue
    if c in protected_zero_cols:
        continue
    if c in clin_knn_set:
        continue
    if X_tr_raw[c].notna().sum() < 20:
        continue
    cont_knn_cols.append(c)

knn_features = sorted(set(cont_knn_cols) | clin_knn_set)

if cont_knn_cols and knn_features:
    print(f"KNN global sur {len(cont_knn_cols)} features continues (distance sur {len(knn_features)} features).")
    knn_global = KNNImputer(
        n_neighbors=7,
        weights="distance",
        metric="nan_euclidean",
    )

    tr_knn = pd.DataFrame(
        knn_global.fit_transform(X_tr_raw[knn_features]),
        index=X_tr_raw.index,
        columns=knn_features,
    )
    te_knn = pd.DataFrame(
        knn_global.transform(X_te_raw[knn_features]),
        index=X_te_raw.index,
        columns=knn_features,
    )

    # On NE TOUCHE PAS aux variables cliniques déjà imputées par le premier KNN
    for c in cont_knn_cols:
        X_tr_raw[c] = tr_knn[c]
        X_te_raw[c] = te_knn[c]
else:
    print("Pas de features supplémentaires pour le KNN global (cont_knn_cols vide).")

# 2) Filet de sécurité : médiane pour les NaN restants
imp = SimpleImputer(strategy="median")
sc  = StandardScaler()

X_tr_imp = pd.DataFrame(
    imp.fit_transform(X_tr_raw),
    index=X_tr_raw.index,
    columns=X_tr_raw.columns,
)
X_te_imp = pd.DataFrame(
    imp.transform(X_te_raw),
    index=X_te_raw.index,
    columns=X_te_raw.columns,
)

X_tr_std = pd.DataFrame(
    sc.fit_transform(X_tr_imp),
    index=X_tr_imp.index,
    columns=X_tr_imp.columns,
)
X_te_std = pd.DataFrame(
    sc.transform(X_te_imp),
    index=X_te_imp.index,
    columns=X_te_imp.columns,
)

vt = VarianceThreshold(1e-5)
X_tr_v = vt.fit_transform(X_tr_std)
X_te_v = vt.transform(X_te_std)
cols_v = X_tr_std.columns[vt.get_support()]

X_tr = pd.DataFrame(X_tr_v, index=X_tr_std.index, columns=cols_v)
X_te = pd.DataFrame(X_te_v, index=X_te_std.index, columns=cols_v)

# drop highly correlated
corr = X_tr.corr().abs()
mask_triu = np.triu(np.ones_like(corr.values, dtype=bool), k=1)
hi_i, hi_j = np.where((corr.values > 0.995) & mask_triu)
drop = sorted(set(corr.columns[j] for j in hi_j))
X_tr.drop(columns=drop, inplace=True, errors="ignore")
X_te = X_te.reindex(columns=X_tr.columns).fillna(0.0)

# clip for Cox / trees stability
X_tr = X_tr.clip(-6, 6).astype("float32")
X_te = X_te.clip(-6, 6).astype("float32")

print(f"Features final: {X_tr.shape[1]}")

# ----------------------------------------------------------------------
# MOLECULAR VIEW (subset of features)
# ----------------------------------------------------------------------
mol_view_cols = [c for c in X_tr.columns if c in mol_tr_agg.columns]
X_tr_mol = X_tr[mol_view_cols].copy()
X_te_mol = X_te[mol_view_cols].copy()
print(f"Mol-view features: {len(mol_view_cols)} (vs {X_tr.shape[1]} total)")

# ----------------------------------------------------------------------
# AFT helpers — Optuna-tuned XGB-AFT
# ----------------------------------------------------------------------
y_time  = pd.Series(y_sks["time"],  index=X_tr.index).astype(float)
y_event = pd.Series(y_sks["event"], index=X_tr.index).astype(int)

def aft_bounds(t, e):
    lower = t.values.copy().astype(float)
    upper = t.values.copy().astype(float)
    upper[e.values == 0] = np.inf
    return lower, upper

# Optuna best params (HEAVY search)
aft_cfg_optuna = dict(
    objective="survival:aft",
    tree_method="hist",
    eta=0.020925286079506073,
    max_depth=4,
    subsample=0.69930442056126,
    colsample_bytree=0.8095383866684407,
    reg_lambda=1.364044904816541,
    reg_alpha=0.5074769655649117,
    aft_loss_distribution="normal",
    aft_loss_sigma=1.0781360476258952,
    min_child_weight=13.548407642780635,
)

# On fait un petit ensemble: mêmes hyperparams, seeds XGB différents
aft_cfgs = [
    ("optuna", aft_cfg_optuna, RNG),
    ("optuna", aft_cfg_optuna, RNG + 7),
    ("optuna", aft_cfg_optuna, RNG + 17),
    ("optuna", aft_cfg_optuna, RNG + 31),
]

def fit_aft_cfg(params, seed, kf):
    oof = np.zeros(len(X_tr))
    tst = np.zeros(len(X_te))
    cfg = params.copy()
    cfg.update(dict(nthread=-1, verbosity=0, seed=seed))

    for tr_idx, va_idx in kf.split(X_tr):
        Xt, Xv = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        t_tr, e_tr = y_time.iloc[tr_idx], y_event.iloc[tr_idx]
        t_va, e_va = y_time.iloc[va_idx], y_event.iloc[va_idx]
        lb_tr, ub_tr = aft_bounds(t_tr, e_tr)
        lb_va, ub_va = aft_bounds(t_va, e_va)

        dtr = xgb.DMatrix(Xt.values)
        dva = xgb.DMatrix(Xv.values)
        dte = xgb.DMatrix(X_te.values)

        dtr.set_float_info("label_lower_bound", lb_tr)
        dtr.set_float_info("label_upper_bound", ub_tr)
        dva.set_float_info("label_lower_bound", lb_va)
        dva.set_float_info("label_upper_bound", ub_va)

        booster = xgb.train(
            params=cfg,
            dtrain=dtr,
            num_boost_round=3500,          # aligné avec l'Optuna heavy
            evals=[(dva, "val")],
            early_stopping_rounds=200,
            verbose_eval=False,
        )

        oof[va_idx] = -booster.predict(
            dva, iteration_range=(0, booster.best_iteration + 1)
        )
        tst += -booster.predict(
            dte, iteration_range=(0, booster.best_iteration + 1)
        ) / N_FOLDS

    c = concordance_index_ipcw(y_sks, y_sks, oof)[0]
    return c, oof, tst


# ----------------------------------------------------------------------
# LGBM helpers (full & mol views)
# ----------------------------------------------------------------------
def fit_lgbm_cfg_view(lgb_params, tag, kf, X_tr_view, X_te_view):
    oof = np.zeros(len(X_tr_view))
    tst = np.zeros(len(X_te_view))

    for tr_idx, va_idx in kf.split(X_tr_view):
        Xt, Xv = X_tr_view.iloc[tr_idx], X_tr_view.iloc[va_idx]
        yt, yv = y_sks[tr_idx], y_sks[va_idx]

        teacher = CoxPHSurvivalAnalysis(alpha=5e-2, n_iter=500, tol=1e-8)
        teacher.fit(Xt, yt)
        r_tr = teacher.predict(Xt)
        r_va = teacher.predict(Xv)

        student = lgb.LGBMRegressor(**lgb_params)
        student.fit(
            Xt, r_tr,
            eval_set=[(Xv, r_va)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(stopping_rounds=260, verbose=False)],
        )

        oof[va_idx] = student.predict(Xv, num_iteration=student.best_iteration_)
        tst += student.predict(X_te_view, num_iteration=student.best_iteration_) / N_FOLDS

    c = concordance_index_ipcw(y_sks, y_sks, oof)[0]
    print(f"OOF IPCW-C -> LGBM({tag}): {c:.4f}")
    return c, oof, tst

# full-view configs (OPTUNA-TUNED MAIN)
# Best LGBM FULL OOF C-index: 0.7164690588622358
lgb_full_cfg_main = dict(
    n_estimators=3726,
    learning_rate=0.04583400134683942,
    num_leaves=70,
    min_data_in_leaf=60,
    feature_fraction=0.5778221450147977,
    bagging_fraction=0.8719527732639356,
    bagging_freq=1,
    reg_lambda=0.5693457391382313,
    reg_alpha=0.10267810192669191,
    random_state=RNG,
    n_jobs=-1,
    verbose=-1,
)

# on garde les 2 autres configs originales pour diversité
lgb_full_cfg_strong = dict(
    n_estimators=5200,
    learning_rate=0.03,
    num_leaves=56,
    min_data_in_leaf=90,
    feature_fraction=0.62,
    bagging_fraction=0.78,
    bagging_freq=1,
    reg_lambda=2.5,
    reg_alpha=0.0,
    random_state=RNG + 7,
    n_jobs=-1,
    verbose=-1,
)

lgb_full_cfg_extra = dict(
    n_estimators=5200,
    learning_rate=0.03,
    num_leaves=40,
    min_data_in_leaf=110,
    feature_fraction=0.58,
    bagging_fraction=0.78,
    bagging_freq=1,
    reg_lambda=3.0,
    reg_alpha=0.0,
    random_state=RNG + 21,
    n_jobs=-1,
    verbose=-1,
)

lgb_full_configs = [
    ("full_main_opt",   lgb_full_cfg_main),
    ("full_strong_reg", lgb_full_cfg_strong),
    ("full_extra_reg",  lgb_full_cfg_extra),
]


# mol-view configs
lgb_mol_cfg_main = dict(
    n_estimators=4300,
    learning_rate=0.03,
    num_leaves=56,
    min_data_in_leaf=50,
    feature_fraction=0.78,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=1.8,
    reg_alpha=0.0,
    random_state=RNG + 101,
    n_jobs=-1,
    verbose=-1,
)

lgb_mol_cfg_strong = dict(
    n_estimators=4300,
    learning_rate=0.03,
    num_leaves=42,
    min_data_in_leaf=60,
    feature_fraction=0.73,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=2.4,
    reg_alpha=0.0,
    random_state=RNG + 137,
    n_jobs=-1,
    verbose=-1,
)

lgb_mol_configs = [
    ("mol_main",   lgb_mol_cfg_main),
    ("mol_strong", lgb_mol_cfg_strong),
]

# ----------------------------------------------------------------------
# TRAINING WITH CV-BAGGING OVER CV_SEEDS
# ----------------------------------------------------------------------
aft_oof_seeds = []
aft_tst_seeds = []

lgb_full_oof_seeds = []
lgb_full_tst_seeds = []

lgb_mol_oof_seeds = []
lgb_mol_tst_seeds = []

for cv_seed in CV_SEEDS:
    print(f"\n=== CV seed {cv_seed} ===")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)

    # ---- XGB-AFT (4 modèles) pour ce cv_seed ----
    aft_seed_oofs = []
    aft_seed_tsts = []
    for name, cfg, s in aft_cfgs:
        print(f"\n>>> Training AFT {name}_seed{s-RNG} (cv_seed={cv_seed})")
        c, oof_c, tst_c = fit_aft_cfg(cfg, s, kf)
        print(f"OOF IPCW-C -> AFT({name}) seed{s-RNG}: {c:.4f}")
        aft_seed_oofs.append(oof_c)
        aft_seed_tsts.append(tst_c)

    oof_aft_seed = np.mean(aft_seed_oofs, axis=0)
    tst_aft_seed = np.mean(aft_seed_tsts, axis=0)
    c_aft_seed = concordance_index_ipcw(y_sks, y_sks, oof_aft_seed)[0]
    print(f"OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed={cv_seed}]: {c_aft_seed:.4f}")

    aft_oof_seeds.append(oof_aft_seed)
    aft_tst_seeds.append(tst_aft_seed)

    # ---- LGBM full-view pour ce cv_seed ----
    full_oofs = []
    full_tsts = []
    full_scores = []

    print(f"\n>>> Training LGBM (full view, cv_seed={cv_seed})")
    for tag, cfg in lgb_full_configs:
        c, oof_c, tst_c = fit_lgbm_cfg_view(cfg, tag, kf, X_tr, X_te)
        full_oofs.append(oof_c)
        full_tsts.append(tst_c)
        full_scores.append(c)

    weights_full = np.array(full_scores)
    if weights_full.sum() <= 0:
        weights_full = np.ones_like(weights_full) / len(weights_full)
    else:
        weights_full = weights_full / weights_full.sum()

    oof_full_seed = np.average(full_oofs, axis=0, weights=weights_full)
    tst_full_seed = np.average(full_tsts, axis=0, weights=weights_full)
    c_full_seed = concordance_index_ipcw(y_sks, y_sks, oof_full_seed)[0]
    print(f"OOF IPCW-C -> LGBM ensemble(3, full view) [cv_seed={cv_seed}]: {c_full_seed:.4f}")
    print("LGBM full-view ensemble weights:",
          dict(zip([t for t,_ in lgb_full_configs], weights_full)))

    lgb_full_oof_seeds.append(oof_full_seed)
    lgb_full_tst_seeds.append(tst_full_seed)

    # ---- LGBM mol-view pour ce cv_seed ----
    mol_oofs = []
    mol_tsts = []
    mol_scores = []

    print(f"\n>>> Training LGBM (molecular view, cv_seed={cv_seed})")
    for tag, cfg in lgb_mol_configs:
        c, oof_c, tst_c = fit_lgbm_cfg_view(cfg, f"{tag}, mol-view", kf, X_tr_mol, X_te_mol)
        mol_oofs.append(oof_c)
        mol_tsts.append(tst_c)
        mol_scores.append(c)

    weights_mol = np.array(mol_scores)
    if weights_mol.sum() <= 0:
        weights_mol = np.ones_like(weights_mol) / len(weights_mol)
    else:
        weights_mol = weights_mol / weights_mol.sum()

    oof_mol_seed = np.average(mol_oofs, axis=0, weights=weights_mol)
    tst_mol_seed = np.average(mol_tsts, axis=0, weights=weights_mol)
    c_mol_seed = concordance_index_ipcw(y_sks, y_sks, oof_mol_seed)[0]
    print(f"OOF IPCW-C -> LGBM ensemble(2, mol view) [cv_seed={cv_seed}]: {c_mol_seed:.4f}")
    print("LGBM mol-view ensemble weights:",
          dict(zip([t for t,_ in lgb_mol_configs], weights_mol)))

    lgb_mol_oof_seeds.append(oof_mol_seed)
    lgb_mol_tst_seeds.append(tst_mol_seed)

# ----------------------------------------------------------------------
# CV-BAGGING ACROSS SEEDS
# ----------------------------------------------------------------------
oof_aft_cvbag = np.mean(aft_oof_seeds, axis=0)
tst_aft_cvbag = np.mean(aft_tst_seeds, axis=0)

oof_lgbm_full_cvbag = np.mean(lgb_full_oof_seeds, axis=0)
tst_lgbm_full_cvbag = np.mean(lgb_full_tst_seeds, axis=0)

oof_lgbm_mol_cvbag = np.mean(lgb_mol_oof_seeds, axis=0)
tst_lgbm_mol_cvbag = np.mean(lgb_mol_tst_seeds, axis=0)

c_aft_cvbag   = concordance_index_ipcw(y_sks, y_sks, oof_aft_cvbag)[0]
c_full_cvbag  = concordance_index_ipcw(y_sks, y_sks, oof_lgbm_full_cvbag)[0]
c_mol_cvbag   = concordance_index_ipcw(y_sks, y_sks, oof_lgbm_mol_cvbag)[0]
print(f"\nOOF IPCW-C -> XGB-AFT ensemble(4) + CV-bagging: {c_aft_cvbag:.4f}")
print(f"OOF IPCW-C -> LGBM full ensemble(3) + CV-bagging: {c_full_cvbag:.4f}")
print(f"OOF IPCW-C -> LGBM mol-view ensemble(2) + CV-bagging: {c_mol_cvbag:.4f}")

# ----------------------------------------------------------------------
# RANKS + BLENDS (G/H/I + FINAL 0.57/0.28/0.15)
# ----------------------------------------------------------------------
o_x  = to_rank(oof_aft_cvbag)
o_lf = to_rank(oof_lgbm_full_cvbag)
o_lm = to_rank(oof_lgbm_mol_cvbag)

t_x  = to_rank(tst_aft_cvbag)
t_lf = to_rank(tst_lgbm_full_cvbag)
t_lm = to_rank(tst_lgbm_mol_cvbag)

print("OOF IPCW-C -> AFT only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_x)[0])
print("OOF IPCW-C -> LGBM full only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_lf)[0])
print("OOF IPCW-C -> LGBM mol-view only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_lm)[0])

blend_configs = {
    "FINAL": (0.57, 0.28, 0.15),   # ton blend optimisé
}

for name, (w_x, w_lf, w_lm) in blend_configs.items():
    w_sum = w_x + w_lf + w_lm
    if not np.isclose(w_sum, 1.0):
        print(f"[WARN] Blends {name}: weights do not sum to 1 (sum={w_sum:.4f}), renormalising.")
        w_x  /= w_sum
        w_lf /= w_sum
        w_lm /= w_sum

    o_blend = w_x * o_x + w_lf * o_lf + w_lm * o_lm
    c_blend = concordance_index_ipcw(y_sks, y_sks, o_blend)[0]
    print(f"OOF IPCW-C -> Blends {name} (w_x={w_x:.2f}, w_lf={w_lf:.2f}, w_lm={w_lm:.2f}): {c_blend:.4f}")

    t_blend = w_x * t_x + w_lf * t_lf + w_lm * t_lm
    out_name = f"submission_twin_v2_22_blends_{name}_wx{w_x:.2f}_wlf{w_lf:.2f}_wlm{w_lm:.2f}.csv"
    pd.DataFrame(
        {"ID": X_te.index, "risk_score": t_blend}
    ).set_index("ID").to_csv(out_name)
    print(f"✅ Wrote: {out_name}")

# also write raw heads
pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_x}
).set_index("ID").to_csv("submission_twin_v2_22_aft_ensemble_cvbag.csv")

pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_lf}
).set_index("ID").to_csv("submission_twin_v2_22_lgbm_full_ens3_cvbag_rank.csv")

pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_lm}
).set_index("ID").to_csv("submission_twin_v2_22_lgbm_mol_ens2_cvbag_rank.csv")

print("Wrote base heads + FINAL blends (v2.22 + KNN clinique + KNN global).")
