
# ===================== QRT — Twin V3.8 (AML-informed FE + 3-head blend) =====================
import os, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
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
    for base in ("../data", "./"):
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

def normalize_effect(s):
    # bien distinguer non_synonymous / missense / truncs / splice / synon
    s = str(s).lower()
    if "non_synonymous" in s or "non-synonymous" in s or "missense" in s:
        return "missense_like"
    if "frame" in s:
        return "frameshift"
    if "stop" in s:
        return "nonsense"
    if "splice" in s:
        return "splice"
    if "synonymous" in s:
        return "synonymous"
    if "itd" in s:
        return "itd"
    return "other"

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
# CYTO FEATURES (base + complexity extra)
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

def parse_cyto_base(s: str):
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

def parse_cyto_extra(s: str):
    t = str(s).lower() if pd.notnull(s) else ""
    if not t:
        return pd.Series({
            "n_translocations": 0,
            "n_deletions": 0,
            "n_trisomies": 0,
            "n_monosomies": 0,
            "monosomal_karyotype": 0,
        })
    n_trans = len(re.findall(r"t\(", t))
    n_del   = len(re.findall(r"del\(", t))
    n_tri   = len(re.findall(r"\+\d+", t))
    n_mono  = len(re.findall(r"-\d+", t))
    mono_karyotype = int((n_mono >= 2) or (n_mono >= 1 and (n_del + n_tri) >= 1))
    return pd.Series({
        "n_translocations": n_trans,
        "n_deletions": n_del,
        "n_trisomies": n_tri,
        "n_monosomies": n_mono,
        "monosomal_karyotype": mono_karyotype,
    })

for df in (clin_tr, clin_te):
    cy_base  = df["CYTOGENETICS"].apply(parse_cyto_base)
    cy_extra = df["CYTOGENETICS"].apply(parse_cyto_extra)
    for c in cy_base.columns:
        df[c] = cy_base[c]
    for c in cy_extra.columns:
        df[c] = cy_extra[c]

# ----------------------------------------------------------------------
# CLINICAL TRANSFORMS + MISSINGNESS + SEVERITY
# ----------------------------------------------------------------------
CLIN_CONT = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]

# stats sur le train (raw) pour les flags de sévérité
q_stats = {}
for c in CLIN_CONT:
    tr = clin_tr[c]
    q_stats[c] = {
        "q10": float(tr.quantile(0.10)),
        "q25": float(tr.quantile(0.25)),
        "q75": float(tr.quantile(0.75)),
        "q90": float(tr.quantile(0.90)),
    }

for df in (clin_tr, clin_te):
    # garder une copie raw pour les flags
    for c in CLIN_CONT:
        df[f"{c}_raw"] = df[c]

    # missingness flags
    for c in CLIN_CONT:
        df[f"MISS_{c}"] = df[c].isna().astype(int)

    # log1p sur les principaux
    for c in ["WBC", "ANC", "MONOCYTES", "PLT", "BM_BLAST"]:
        df[c] = np.log1p(df[c])

    # winsorisation (sur l'échelle loggée)
    for c in CLIN_CONT:
        df[c] = winsorize(df[c])

    # ratios
    df["NEUT_RATIO"] = df["ANC"] / (df["WBC"] + 1e-6)
    df["MONO_RATIO"] = df["MONOCYTES"] / (df["WBC"] + 1e-6)

    # proxy de charge tumorale (sur raw)
    df["BLAST_WBC_BURDEN"] = df["BM_BLAST_raw"].fillna(0.0) * df["WBC_raw"].fillna(0.0)

    # flags de sévérité (sur raw)
    df["HB_LOW"]        = (df["HB_raw"]  <= q_stats["HB"]["q10"]).astype(int)
    df["HB_HIGH"]       = (df["HB_raw"]  >= q_stats["HB"]["q90"]).astype(int)
    df["PLT_LOW"]       = (df["PLT_raw"] <= q_stats["PLT"]["q10"]).astype(int)
    df["PLT_HIGH"]      = (df["PLT_raw"] >= q_stats["PLT"]["q90"]).astype(int)
    df["WBC_HIGH"]      = (df["WBC_raw"] >= q_stats["WBC"]["q90"]).astype(int)
    df["BM_BLAST_HIGH"] = (df["BM_BLAST_raw"] >= q_stats["BM_BLAST"]["q75"]).astype(int)

# ----------------------------------------------------------------------
# MOLECULAR PREPROCESSING
# ----------------------------------------------------------------------
for df in (mol_tr, mol_te):
    df["EFFECT_N"] = df["EFFECT"].map(normalize_effect)
    df["VAF"]      = pd.to_numeric(df["VAF"], errors="coerce").clip(0, 1)
    df["DEPTH"]    = pd.to_numeric(df.get("DEPTH", np.nan), errors="coerce")
    df["GENE"]     = df["GENE"].astype(str)

# curated gene lists
CURATED = [
    "NPM1","FLT3","DNMT3A","TET2","ASXL1","RUNX1",
    "IDH1","IDH2","TP53","SRSF2","SF3B1","U2AF1",
    "NRAS","KRAS","CBL","PTPN11","KIT","CEBPA"
]
MR_GENES = ["ASXL1","BCOR","EZH2","RUNX1","SF3B1","SRSF2","STAG2","U2AF1","ZRSR2"]
CURATED = sorted(set(CURATED + MR_GENES))

PATH_RAS = {"NRAS", "KRAS", "PTPN11", "CBL", "NF1"}
PATH_EPI = {"DNMT3A", "TET2", "ASXL1", "IDH1", "IDH2", "EZH2"}
PATH_SPL = {"SRSF2", "SF3B1", "U2AF1", "ZRSR2", "STAG2"}

# top genes pour VAFmax par gène
gene_freq_all = pd.concat([mol_tr["GENE"], mol_te["GENE"]]).value_counts()
TOP_N_GENES = 80
top_genes = set(gene_freq_all.head(TOP_N_GENES).index) | set(CURATED)

def aggregate_mol(df):
    g = df.groupby("ID")

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

    slopes = g["VAF"].apply(vaf_slopes_series)

    n_ge_035 = g.apply(lambda x: (x["VAF"] >= 0.35).sum()).rename("n_mut_vaf_ge_035")
    n_ge_045 = g.apply(lambda x: (x["VAF"] >= 0.45).sum()).rename("n_mut_vaf_ge_045")

    # subclonal features
    def subclonal_stats(group):
        v = group["VAF"].dropna().values
        if v.size == 0:
            return pd.Series({
                "subclonal_burden": np.nan,
                "n_subclonal": 0,
                "n_clonal": 0,
                "subclonal_fraction": np.nan,
            })
        v_sorted = np.sort(v)[::-1]
        v_max = v_sorted[0]
        subclonal = (v < 0.3).sum()
        clonal = (v >= 0.4).sum()
        return pd.Series({
            "subclonal_burden": float(v.sum() - v_max),
            "n_subclonal": int(subclonal),
            "n_clonal": int(clonal),
            "subclonal_fraction": float(subclonal / len(v)),
        })
    subcl = g.apply(subclonal_stats)

    # effect counts global
    eff = df.pivot_table(
        index="ID", columns="EFFECT_N", values="VAF", aggfunc="count"
    ).fillna(0.0)
    eff.columns = [f"eff_{c}" for c in eff.columns]

    # curated gene presence
    pres = df[df["GENE"].isin(CURATED)].pivot_table(
        index="ID", columns="GENE", values="VAF", aggfunc="max"
    )
    if pres.size > 0:
        pres = (pres.notna()).astype(int)
        pres.columns = [f"has__{c}" for c in pres.columns]
    else:
        pres = pd.DataFrame(index=base.index)

    # per-gene VAFmax
    df_top = df[df["GENE"].isin(top_genes)]
    by_gene = df_top.groupby(["ID", "GENE"]).agg(VAF_max=("VAF", "max")).reset_index()
    if not by_gene.empty:
        vaf_max = by_gene.pivot(index="ID", columns="GENE", values="VAF_max")
        vaf_max.columns = [f"VAFmax__{g}" for g in vaf_max.columns]
    else:
        vaf_max = pd.DataFrame(index=base.index)

    # pathway stats (count, VAFmax, VAFsum)
    def path_stats(group, path_genes):
        mask = group["GENE"].isin(path_genes)
        if not mask.any():
            return pd.Series({"count": 0, "VAFmax": np.nan, "VAFsum": 0.0})
        v = group.loc[mask, "VAF"].dropna()
        if v.empty:
            return pd.Series({"count": int(mask.sum()), "VAFmax": np.nan, "VAFsum": 0.0})
        return pd.Series({"count": int(mask.sum()), "VAFmax": float(v.max()), "VAFsum": float(v.sum())})

    path_ras = g.apply(lambda x: path_stats(x, PATH_RAS))
    path_ras.columns = [f"path_RAS_{c}" for c in path_ras.columns]
    path_epi = g.apply(lambda x: path_stats(x, PATH_EPI))
    path_epi.columns = [f"path_EPI_{c}" for c in path_epi.columns]
    path_spl = g.apply(lambda x: path_stats(x, PATH_SPL))
    path_spl.columns = [f"path_SPL_{c}" for c in path_spl.columns]

    # FLT3-ITD specific
    def flt3_itd_stats(group):
        mask_itd = (group["GENE"] == "FLT3") & (group["EFFECT"].str.contains("ITD", case=False, na=False))
        if not mask_itd.any():
            return pd.Series({"has__FLT3_ITD": 0, "FLT3_ITD_VAF_max": np.nan, "FLT3_ITD_count": 0})
        v = group.loc[mask_itd, "VAF"].dropna()
        return pd.Series({
            "has__FLT3_ITD": 1,
            "FLT3_ITD_VAF_max": float(v.max()) if not v.empty else np.nan,
            "FLT3_ITD_count": int(mask_itd.sum()),
        })
    flt3_itd = g.apply(flt3_itd_stats)

    # TSG / ONCO tiers
    TSG_GENES  = ["TP53", "RUNX1", "ASXL1", "STAG2", "EZH2"]
    ONCO_GENES = ["NRAS", "KRAS", "FLT3", "KIT", "CBL", "PTPN11"]

    def tier_stats(group, genes):
        sub = group[group["GENE"].isin(genes)]
        if sub.empty:
            return pd.Series({
                "truncating_any": 0,
                "missense_any": 0,
                "truncating_count": 0,
                "missense_count": 0,
            })
        effs = sub["EFFECT_N"]
        trunc = effs.isin(["frameshift", "nonsense", "splice"]).sum()
        miss  = effs.isin(["missense_like", "itd"]).sum()
        return pd.Series({
            "truncating_any": int(trunc > 0),
            "missense_any": int(miss > 0),
            "truncating_count": int(trunc),
            "missense_count": int(miss),
        })

    tsg_tier  = g.apply(lambda x: tier_stats(x, TSG_GENES))
    tsg_tier.columns = [f"TSG_{c}" for c in tsg_tier.columns]
    onco_tier = g.apply(lambda x: tier_stats(x, ONCO_GENES))
    onco_tier.columns = [f"ONCO_{c}" for c in onco_tier.columns]

    # assemble
    agg = base.join(
        [
            vaf_gini,
            vaf_top3_mean,
            prop_vaf_lt_005,
            prop_vaf_ge_035,
            slopes,
            n_ge_035,
            n_ge_045,
            subcl,
            eff,
            pres,
            vaf_max,
            path_ras,
            path_epi,
            path_spl,
            flt3_itd,
            tsg_tier,
            onco_tier,
        ],
        how="left",
    )

    # derived scores
    agg["clonal_potential"] = agg["n_vars"] * agg["VAF_sum"]

    tp53 = agg.get("has__TP53", 0)
    ras_count  = agg.get("path_RAS_count", 0)
    epi_count  = agg.get("path_EPI_count", 0)

    agg["genomic_risk_score"] = 2 * tp53 + ras_count + epi_count

    # MR burden
    mr_cols = [c for c in agg.columns if c.startswith("has__") and c[5:] in MR_GENES]
    if mr_cols:
        agg["mr_gene_mut_count"] = agg[mr_cols].sum(axis=1)
        agg["mr_gene_mut_any"]   = (agg["mr_gene_mut_count"] > 0).astype(int)
    else:
        agg["mr_gene_mut_count"] = 0
        agg["mr_gene_mut_any"]   = 0

    # NPM1-ish favorable profile (approx)
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
    agg["adverse_mol_flag"]    = ((has_tp53 == 1) | (mr_any == 1)).astype(int)

    return agg

mol_tr_agg = aggregate_mol(mol_tr)
mol_te_agg = aggregate_mol(mol_te)
print("Mol_agg shapes:", mol_tr_agg.shape, mol_te_agg.shape)

# ----------------------------------------------------------------------
# BUILD X + ELN-LIKE RISK
# ----------------------------------------------------------------------
def build_matrix(clin, mol_agg):
    X = clin.set_index("ID").join(mol_agg, how="left")

    adv_cyto_cols = [
        "monosomy7", "del7q", "monosomy5", "del5q",
        "inv3_t3_3", "del20q", "complex_karyotype", "monosomal_karyotype"
    ]
    fav_cyto_cols = ["t_8_21", "inv16_t16", "t_15_17"]

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
    X["eln_like_numeric"]      = eln_fav * (-1) + eln_adv * (1)

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
# CENTER one-hot encoding (comme v2.15)
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

# d'abord on drop les colonnes complètement vides
X_tr_raw.drop(columns=na_all_train, inplace=True)
X_te_raw.drop(columns=na_all_train, inplace=True)

# puis on recalcule few_data_cols APRÈS ce drop
few_data_cols = X_tr_raw.columns[(X_tr_raw.notna().sum() < 5)]

# et on drop en mode "ignore" pour éviter les KeyError si jamais
X_tr_raw.drop(columns=few_data_cols, inplace=True, errors="ignore")
X_te_raw.drop(columns=few_data_cols, inplace=True, errors="ignore")

# drop raw text cyto only
for df in (X_tr_raw, X_te_raw):
    if "CYTOGENETICS" in df.columns:
        df.drop(columns="CYTOGENETICS", inplace=True)

print(f"Features before clean: {len(all_cols)}")
print(f"Features final (before scaling): {X_tr_raw.shape[1]}")

# ----------------------------------------------------------------------
# IMPUTE + SCALE + VARIANCE FILTER + DROP HIGH CORR
# ----------------------------------------------------------------------
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

# clip pour stabilité
X_tr = X_tr.clip(-6, 6).astype("float32")
X_te = X_te.clip(-6, 6).astype("float32")

print(f"Features final: {X_tr.shape[1]}")

# ----------------------------------------------------------------------
# MOLECULAR VIEW (subset of features provenant de mol_agg)
# ----------------------------------------------------------------------
mol_view_cols = [c for c in X_tr.columns if c in mol_tr_agg.columns]
X_tr_mol = X_tr[mol_view_cols].copy()
X_te_mol = X_te[mol_view_cols].copy()
print(f"Mol-view features: {len(mol_view_cols)} (vs {X_tr.shape[1]} total)")

# ----------------------------------------------------------------------
# AFT helpers
# ----------------------------------------------------------------------
y_time  = pd.Series(y_sks["time"],  index=X_tr.index).astype(float)
y_event = pd.Series(y_sks["event"], index=X_tr.index).astype(int)

def aft_bounds(t, e):
    lower = t.values.copy().astype(float)
    upper = t.values.copy().astype(float)
    upper[e.values == 0] = np.inf
    return lower, upper

# hyperparams très proches de v2.15
aft_cfg_base = dict(
    objective="survival:aft",
    tree_method="hist",
    eta=0.045,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.7,
    reg_lambda=1.2,
    reg_alpha=0.0,
    aft_loss_distribution="normal",
    aft_loss_sigma=0.6,
)
aft_cfg_alt = dict(
    objective="survival:aft",
    tree_method="hist",
    eta=0.045,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.7,
    reg_lambda=1.2,
    reg_alpha=0.0,
    aft_loss_distribution="normal",
    aft_loss_sigma=0.8,
)

aft_cfgs = [
    ("base", aft_cfg_base, RNG),
    ("base", aft_cfg_base, RNG + 17),
    ("alt",  aft_cfg_alt,  RNG),
    ("alt",  aft_cfg_alt,  RNG + 17),
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
            num_boost_round=3500,
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
# LGBM helpers (full & mol views) -- student-of-Cox comme v2.15
# ----------------------------------------------------------------------
def fit_lgbm_cfg_view(lgb_params, tag, kf, X_tr_view, X_te_view):
    oof = np.zeros(len(X_tr_view))
    tst = np.zeros(len(X_te_view))

    for tr_idx, va_idx in kf.split(X_tr_view):
        Xt, Xv = X_tr_view.iloc[tr_idx], X_tr_view.iloc[va_idx]
        yt, yv = y_sks[tr_idx], y_sks[va_idx]

        teacher = CoxPHSurvivalAnalysis(alpha=1e-2, n_iter=800, tol=1e-9)
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

# full-view configs (identiques à v2.15)
lgb_full_cfg_main = dict(
    n_estimators=5500,
    learning_rate=0.03,
    num_leaves=80,
    min_data_in_leaf=60,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=1.5,
    reg_alpha=0.0,
    random_state=RNG,
    n_jobs=-1,
    verbose=-1,
)
lgb_full_cfg_strong = dict(
    n_estimators=5500,
    learning_rate=0.03,
    num_leaves=64,
    min_data_in_leaf=70,
    feature_fraction=0.65,
    bagging_fraction=0.75,
    bagging_freq=1,
    reg_lambda=2.0,
    reg_alpha=0.0,
    random_state=RNG + 7,
    n_jobs=-1,
    verbose=-1,
)
lgb_full_cfg_extra = dict(
    n_estimators=5500,
    learning_rate=0.03,
    num_leaves=48,
    min_data_in_leaf=80,
    feature_fraction=0.60,
    bagging_fraction=0.75,
    bagging_freq=1,
    reg_lambda=2.5,
    reg_alpha=0.0,
    random_state=RNG + 21,
    n_jobs=-1,
    verbose=-1,
)

lgb_full_configs = [
    ("full_main",       lgb_full_cfg_main),
    ("full_strong_reg", lgb_full_cfg_strong),
    ("full_extra_reg",  lgb_full_cfg_extra),
]

# mol-view configs (un peu plus petits)
lgb_mol_cfg_main = dict(
    n_estimators=4500,
    learning_rate=0.03,
    num_leaves=64,
    min_data_in_leaf=40,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=1.5,
    reg_alpha=0.0,
    random_state=RNG + 101,
    n_jobs=-1,
    verbose=-1,
)
lgb_mol_cfg_strong = dict(
    n_estimators=4500,
    learning_rate=0.03,
    num_leaves=48,
    min_data_in_leaf=50,
    feature_fraction=0.75,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=2.0,
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
# TRAINING WITH CV-BAGGING OVER CV_SEEDS (3 heads seulement)
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
    print(f"\n>>> Training XGB-AFT (UNIFORM)")
    for name, cfg, s in aft_cfgs:
        print(f"\n>>> AFT {name}_seed{s-RNG} [UNIFORM]")
        c, oof_c, tst_c = fit_aft_cfg(cfg, s, kf)
        print(f"OOF IPCW-C -> AFT({name}, UNIFORM) seed{s-RNG}: {c:.4f}")
        aft_seed_oofs.append(oof_c)
        aft_seed_tsts.append(tst_c)

    oof_aft_seed = np.mean(aft_seed_oofs, axis=0)
    tst_aft_seed = np.mean(aft_seed_tsts, axis=0)
    c_aft_seed = concordance_index_ipcw(y_sks, y_sks, oof_aft_seed)[0]
    print(f"OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed={cv_seed}]: {c_aft_seed:.4f}")

    aft_oof_seeds.append(oof_aft_seed)
    aft_tst_seeds.append(tst_aft_seed)

    # ---- LGBM full-view ----
    full_oofs = []
    full_tsts = []
    full_scores = []

    print(f"\n>>> Training LGBM (full view, UNIFORM, cv_seed={cv_seed})")
    for tag, cfg in lgb_full_configs:
        c, oof_c, tst_c = fit_lgbm_cfg_view(cfg, f"{tag}, full-view", kf, X_tr, X_te)
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

    # ---- LGBM mol-view ----
    mol_oofs = []
    mol_tsts = []
    mol_scores = []

    print(f"\n>>> Training LGBM (molecular view, UNIFORM, cv_seed={cv_seed})")
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
# RANKS + 3-HEAD BLENDS (AFT + full + mol)
# ----------------------------------------------------------------------
o_x  = to_rank(oof_aft_cvbag)
o_lf = to_rank(oof_lgbm_full_cvbag)
o_lm = to_rank(oof_lgbm_mol_cvbag)

t_x  = to_rank(tst_aft_cvbag)
t_lf = to_rank(tst_lgbm_full_cvbag)
t_lm = to_rank(tst_lgbm_mol_cvbag)

print("\n=== OOF C-index (ranks) par head ===")
print("OOF IPCW-C -> AFT only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_x)[0])
print("OOF IPCW-C -> LGBM full only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_lf)[0])
print("OOF IPCW-C -> LGBM mol-view only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_lm)[0])

# Blends type G/H/I (3 têtes)
blend_configs = {
    "G": (0.57, 0.25, 0.18),  # w_x, w_l_full, w_l_mol (v2.15-style)
    "H": (0.50, 0.30, 0.20),
    "I": (0.58, 0.22, 0.20),
}

best_name = None
best_c    = -1
best_w    = None

for name, (w_x, w_lf, w_lm) in blend_configs.items():
    w_sum = w_x + w_lf + w_lm
    if not np.isclose(w_sum, 1.0):
        w_x  /= w_sum
        w_lf /= w_sum
        w_lm /= w_sum

    o_blend = w_x * o_x + w_lf * o_lf + w_lm * o_lm
    c_blend = concordance_index_ipcw(y_sks, y_sks, o_blend)[0]
    print(f"OOF IPCW-C -> Blend {name} (w_x={w_x:.2f}, w_lf={w_lf:.2f}, w_lm={w_lm:.2f}): {c_blend:.4f}")

    t_blend = w_x * t_x + w_lf * t_lf + w_lm * t_lm
    out_name = f"submission_twin_v3_8_blend_{name}_wx{w_x:.2f}_wlf{w_lf:.2f}_wlm{w_lm:.2f}.csv"
    pd.DataFrame(
        {"ID": X_te.index, "risk_score": t_blend}
    ).set_index("ID").to_csv(out_name)
    print(f"✅ Wrote: {out_name}")

    if c_blend > best_c:
        best_c    = c_blend
        best_name = name
        best_w    = (w_x, w_lf, w_lm)

print(f"\nBest 3-head blend by OOF: {best_name} with C={best_c:.4f}, weights={best_w}")

# also write raw heads (rankés)
pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_x}
).set_index("ID").to_csv("submission_twin_v3_8_aft_ensemble_cvbag.csv")

pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_lf}
).set_index("ID").to_csv("submission_twin_v3_8_lgbm_full_ens3_cvbag_rank.csv")

pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_lm}
).set_index("ID").to_csv("submission_twin_v3_8_lgbm_mol_ens2_cvbag_rank.csv")

print("✅ Wrote base heads + G/H/I blends. Twin v3.8 finished.")
=== SHAPES RAW ===
clinical_train: (3323, 9)  clinical_test: (1193, 9)
molecular_train: (10935, 11)  molecular_test: (3089, 11)
target_train: (3323, 3)
Mol_agg shapes: (3026, 153) (1054, 134)
Features before clean: 208
Features final (before scaling): 228
Features final: 222
Mol-view features: 150 (vs 222 total)

=== CV seed 42 ===

>>> Training XGB-AFT (UNIFORM)

>>> AFT base_seed0 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed0: 0.7163

>>> AFT base_seed17 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed17: 0.7178

>>> AFT alt_seed0 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed0: 0.7179

>>> AFT alt_seed17 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed17: 0.7171
OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed=42]: 0.7196

>>> Training LGBM (full view, UNIFORM, cv_seed=42)
OOF IPCW-C -> LGBM(full_main, full-view): 0.7132
OOF IPCW-C -> LGBM(full_strong_reg, full-view): 0.7116
OOF IPCW-C -> LGBM(full_extra_reg, full-view): 0.7116
OOF IPCW-C -> LGBM ensemble(3, full view) [cv_seed=42]: 0.7125
LGBM full-view ensemble weights: {'full_main': np.float64(0.333844327277196), 'full_strong_reg': np.float64(0.33307793533205854), 'full_extra_reg': np.float64(0.3330777373907455)}

>>> Training LGBM (molecular view, UNIFORM, cv_seed=42)
OOF IPCW-C -> LGBM(mol_main, mol-view): 0.6718
OOF IPCW-C -> LGBM(mol_strong, mol-view): 0.6720
OOF IPCW-C -> LGBM ensemble(2, mol view) [cv_seed=42]: 0.6721
LGBM mol-view ensemble weights: {'mol_main': np.float64(0.4999057862275554), 'mol_strong': np.float64(0.5000942137724446)}

=== CV seed 141 ===

>>> Training XGB-AFT (UNIFORM)

>>> AFT base_seed0 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed0: 0.7116

>>> AFT base_seed17 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed17: 0.7132

>>> AFT alt_seed0 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed0: 0.7134

>>> AFT alt_seed17 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed17: 0.7124
OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed=141]: 0.7147

>>> Training LGBM (full view, UNIFORM, cv_seed=141)
OOF IPCW-C -> LGBM(full_main, full-view): 0.7108
OOF IPCW-C -> LGBM(full_strong_reg, full-view): 0.7099
OOF IPCW-C -> LGBM(full_extra_reg, full-view): 0.7097
OOF IPCW-C -> LGBM ensemble(3, full view) [cv_seed=141]: 0.7107
LGBM full-view ensemble weights: {'full_main': np.float64(0.33366047283704087), 'full_strong_reg': np.float64(0.3332192544437759), 'full_extra_reg': np.float64(0.33312027271918326)}

>>> Training LGBM (molecular view, UNIFORM, cv_seed=141)
OOF IPCW-C -> LGBM(mol_main, mol-view): 0.6723
OOF IPCW-C -> LGBM(mol_strong, mol-view): 0.6723
OOF IPCW-C -> LGBM ensemble(2, mol view) [cv_seed=141]: 0.6724
LGBM mol-view ensemble weights: {'mol_main': np.float64(0.49998988291277835), 'mol_strong': np.float64(0.5000101170872217)}

=== CV seed 314 ===

>>> Training XGB-AFT (UNIFORM)

>>> AFT base_seed0 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed0: 0.7183

>>> AFT base_seed17 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed17: 0.7167

>>> AFT alt_seed0 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed0: 0.7171

>>> AFT alt_seed17 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed17: 0.7141
OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed=314]: 0.7187

>>> Training LGBM (full view, UNIFORM, cv_seed=314)
OOF IPCW-C -> LGBM(full_main, full-view): 0.7122
OOF IPCW-C -> LGBM(full_strong_reg, full-view): 0.7124
OOF IPCW-C -> LGBM(full_extra_reg, full-view): 0.7120
OOF IPCW-C -> LGBM ensemble(3, full view) [cv_seed=314]: 0.7127
LGBM full-view ensemble weights: {'full_main': np.float64(0.33331983018195155), 'full_strong_reg': np.float64(0.3334379649809003), 'full_extra_reg': np.float64(0.33324220483714817)}

>>> Training LGBM (molecular view, UNIFORM, cv_seed=314)
OOF IPCW-C -> LGBM(mol_main, mol-view): 0.6752
OOF IPCW-C -> LGBM(mol_strong, mol-view): 0.6738
OOF IPCW-C -> LGBM ensemble(2, mol view) [cv_seed=314]: 0.6745
LGBM mol-view ensemble weights: {'mol_main': np.float64(0.5005221984431588), 'mol_strong': np.float64(0.49947780155684124)}

OOF IPCW-C -> XGB-AFT ensemble(4) + CV-bagging: 0.7204
OOF IPCW-C -> LGBM full ensemble(3) + CV-bagging: 0.7144
OOF IPCW-C -> LGBM mol-view ensemble(2) + CV-bagging: 0.6748

=== OOF C-index (ranks) par head ===
OOF IPCW-C -> AFT only (rank): 0.7203522856678741
OOF IPCW-C -> LGBM full only (rank): 0.7144386217240855
OOF IPCW-C -> LGBM mol-view only (rank): 0.674840028394067
OOF IPCW-C -> Blend G (w_x=0.57, w_lf=0.25, w_lm=0.18): 0.7208
✅ Wrote: submission_twin_v3_8_blend_G_wx0.57_wlf0.25_wlm0.18.csv
OOF IPCW-C -> Blend H (w_x=0.50, w_lf=0.30, w_lm=0.20): 0.7202
✅ Wrote: submission_twin_v3_8_blend_H_wx0.50_wlf0.30_wlm0.20.csv
OOF IPCW-C -> Blend I (w_x=0.58, w_lf=0.22, w_lm=0.20): 0.7206
✅ Wrote: submission_twin_v3_8_blend_I_wx0.58_wlf0.22_wlm0.20.csv

Best 3-head blend by OOF: G with C=0.7208, weights=(0.57, 0.25, 0.18)
✅ Wrote base heads + G/H/I blends. Twin v3.8 finished.
# ===================== QRT — Twin V3.9 (v2.15-like + FLT3-ITD + RSF mol) =====================
import os, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

import xgboost as xgb
import lightgbm as lgb

RNG = 42
N_FOLDS = 5
CV_SEEDS = [42, 141, 314]

# ----------------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------------
def rp(path):
    for base in ("../data", "./"):
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
    if "itd" in s:
        return "itd"
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
# CYTO FEATURES (v2.15 style)
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
# CLINICAL TRANSFORMS + MISSINGNESS (v2.15 style)
# ----------------------------------------------------------------------
CLIN_CONT = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]

for df in (clin_tr, clin_te):
    # missingness flags
    for c in CLIN_CONT:
        df[f"MISS_{c}"] = df[c].isna().astype(int)

    # logs
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

# panel AML curated & pathways (v2.15)
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

    # pathways (counts uniquement comme v2.15)
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

    # assemble core (v2.15)
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

    # -------- FLT3-ITD spécifique (ajout minimal) --------
    def flt3_itd_stats(group):
        mask = (group["GENE"] == "FLT3") & group["EFFECT"].str.contains("ITD", case=False, na=False)
        if not mask.any():
            return pd.Series({"has__FLT3_ITD": 0, "FLT3_ITD_VAF_max": np.nan})
        v = group.loc[mask, "VAF"].dropna()
        return pd.Series({
            "has__FLT3_ITD": 1,
            "FLT3_ITD_VAF_max": float(v.max()) if not v.empty else np.nan,
        })

    flt3_itd = g.apply(flt3_itd_stats)
    agg = agg.join(flt3_itd, how="left")

    # clonal_potential & baseline genomic_risk_score (v2.15)
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
print("Mol_agg shapes:", mol_tr_agg.shape, mol_te_agg.shape)

# ----------------------------------------------------------------------
# BUILD X + ELN-LIKE RISK (v2.15 style)
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
# CENTER one-hot encoding (comme v2.15)
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

# drop all-NaN & almost-empty (avec protection erreurs)
na_all_train = X_tr_raw.columns[X_tr_raw.isna().all()]
X_tr_raw.drop(columns=na_all_train, inplace=True)
X_te_raw.drop(columns=na_all_train, inplace=True)

few_data_cols = X_tr_raw.columns[(X_tr_raw.notna().sum() < 5)]
X_tr_raw.drop(columns=few_data_cols, inplace=True, errors="ignore")
X_te_raw.drop(columns=few_data_cols, inplace=True, errors="ignore")

# drop raw text cyto only
for df in (X_tr_raw, X_te_raw):
    if "CYTOGENETICS" in df.columns:
        df.drop(columns="CYTOGENETICS", inplace=True)

print(f"Features before clean: {len(all_cols)}")
print(f"Features final (before scaling): {X_tr_raw.shape[1]}")

# ----------------------------------------------------------------------
# IMPUTE + SCALE + VARIANCE FILTER + DROP HIGH CORR
# ----------------------------------------------------------------------
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

# clip pour stabilité
X_tr = X_tr.clip(-6, 6).astype("float32")
X_te = X_te.clip(-6, 6).astype("float32")

print(f"Features final: {X_tr.shape[1]}")

# ----------------------------------------------------------------------
# MOLECULAR VIEW (subset de mol_agg)
# ----------------------------------------------------------------------
mol_view_cols = [c for c in X_tr.columns if c in mol_tr_agg.columns]
X_tr_mol = X_tr[mol_view_cols].copy()
X_te_mol = X_te[mol_view_cols].copy()
print(f"Mol-view features: {len(mol_view_cols)} (vs {X_tr.shape[1]} total)")

# ----------------------------------------------------------------------
# AFT helpers
# ----------------------------------------------------------------------
y_time  = pd.Series(y_sks["time"],  index=X_tr.index).astype(float)
y_event = pd.Series(y_sks["event"], index=X_tr.index).astype(int)

def aft_bounds(t, e):
    lower = t.values.copy().astype(float)
    upper = t.values.copy().astype(float)
    upper[e.values == 0] = np.inf
    return lower, upper

aft_cfg_base = dict(
    objective="survival:aft",
    tree_method="hist",
    eta=0.045,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.7,
    reg_lambda=1.2,
    reg_alpha=0.0,
    aft_loss_distribution="normal",
    aft_loss_sigma=0.6,
)
aft_cfg_alt = dict(
    objective="survival:aft",
    tree_method="hist",
    eta=0.045,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.7,
    reg_lambda=1.2,
    reg_alpha=0.0,
    aft_loss_distribution="normal",
    aft_loss_sigma=0.8,
)

aft_cfgs = [
    ("base", aft_cfg_base, RNG),
    ("base", aft_cfg_base, RNG + 17),
    ("alt",  aft_cfg_alt,  RNG),
    ("alt",  aft_cfg_alt,  RNG + 17),
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
            num_boost_round=3500,
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

        teacher = CoxPHSurvivalAnalysis(alpha=1e-2, n_iter=800, tol=1e-9)
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

# full-view configs (v2.15)
lgb_full_cfg_main = dict(
    n_estimators=5500,
    learning_rate=0.03,
    num_leaves=80,
    min_data_in_leaf=60,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=1.5,
    reg_alpha=0.0,
    random_state=RNG,
    n_jobs=-1,
    verbose=-1,
)
lgb_full_cfg_strong = dict(
    n_estimators=5500,
    learning_rate=0.03,
    num_leaves=64,
    min_data_in_leaf=70,
    feature_fraction=0.65,
    bagging_fraction=0.75,
    bagging_freq=1,
    reg_lambda=2.0,
    reg_alpha=0.0,
    random_state=RNG + 7,
    n_jobs=-1,
    verbose=-1,
)
lgb_full_cfg_extra = dict(
    n_estimators=5500,
    learning_rate=0.03,
    num_leaves=48,
    min_data_in_leaf=80,
    feature_fraction=0.60,
    bagging_fraction=0.75,
    bagging_freq=1,
    reg_lambda=2.5,
    reg_alpha=0.0,
    random_state=RNG + 21,
    n_jobs=-1,
    verbose=-1,
)

lgb_full_configs = [
    ("full_main",       lgb_full_cfg_main),
    ("full_strong_reg", lgb_full_cfg_strong),
    ("full_extra_reg",  lgb_full_cfg_extra),
]

# mol-view configs (v2.15)
lgb_mol_cfg_main = dict(
    n_estimators=4500,
    learning_rate=0.03,
    num_leaves=64,
    min_data_in_leaf=40,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=1.5,
    reg_alpha=0.0,
    random_state=RNG + 101,
    n_jobs=-1,
    verbose=-1,
)
lgb_mol_cfg_strong = dict(
    n_estimators=4500,
    learning_rate=0.03,
    num_leaves=48,
    min_data_in_leaf=50,
    feature_fraction=0.75,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=2.0,
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
# RSF helper (mol-view, modèle spécialisé moléculaire)
# ----------------------------------------------------------------------
def fit_rsf_mol(kf, X_tr_view, X_te_view, random_state):
    oof = np.zeros(len(X_tr_view))
    tst = np.zeros(len(X_te_view))

    rsf_params = dict(
        n_estimators=600,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )

    for tr_idx, va_idx in kf.split(X_tr_view):
        Xt, Xv = X_tr_view.iloc[tr_idx], X_tr_view.iloc[va_idx]
        yt, yv = y_sks[tr_idx], y_sks[va_idx]

        rsf = RandomSurvivalForest(**rsf_params)
        rsf.fit(Xt, yt)
        # predict renvoie un score de risque (plus grand = plus à risque)
        oof[va_idx] = rsf.predict(Xv)
        tst += rsf.predict(X_te_view) / N_FOLDS

    c = concordance_index_ipcw(y_sks, y_sks, oof)[0]
    print(f"OOF IPCW-C -> RSF(mol-view): {c:.4f}")
    return c, oof, tst

# ----------------------------------------------------------------------
# TRAINING WITH CV-BAGGING OVER CV_SEEDS
# ----------------------------------------------------------------------
aft_oof_seeds = []
aft_tst_seeds = []

lgb_full_oof_seeds = []
lgb_full_tst_seeds = []

lgb_mol_oof_seeds = []
lgb_mol_tst_seeds = []

rsf_mol_oof_seeds = []
rsf_mol_tst_seeds = []

for cv_seed in CV_SEEDS:
    print(f"\n=== CV seed {cv_seed} ===")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)

    # ---- XGB-AFT (4 modèles) ----
    aft_seed_oofs = []
    aft_seed_tsts = []
    print(f"\n>>> Training XGB-AFT (UNIFORM)")
    for name, cfg, s in aft_cfgs:
        print(f"\n>>> AFT {name}_seed{s-RNG} [UNIFORM]")
        c, oof_c, tst_c = fit_aft_cfg(cfg, s, kf)
        print(f"OOF IPCW-C -> AFT({name}, UNIFORM) seed{s-RNG}: {c:.4f}")
        aft_seed_oofs.append(oof_c)
        aft_seed_tsts.append(tst_c)

    oof_aft_seed = np.mean(aft_seed_oofs, axis=0)
    tst_aft_seed = np.mean(aft_seed_tsts, axis=0)
    c_aft_seed = concordance_index_ipcw(y_sks, y_sks, oof_aft_seed)[0]
    print(f"OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed={cv_seed}]: {c_aft_seed:.4f}")

    aft_oof_seeds.append(oof_aft_seed)
    aft_tst_seeds.append(tst_aft_seed)

    # ---- LGBM full-view ----
    full_oofs = []
    full_tsts = []
    full_scores = []

    print(f"\n>>> Training LGBM (full view, UNIFORM, cv_seed={cv_seed})")
    for tag, cfg in lgb_full_configs:
        c, oof_c, tst_c = fit_lgbm_cfg_view(cfg, f"{tag}, full-view", kf, X_tr, X_te)
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

    # ---- LGBM mol-view ----
    mol_oofs = []
    mol_tsts = []
    mol_scores = []

    print(f"\n>>> Training LGBM (molecular view, UNIFORM, cv_seed={cv_seed})")
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

    # ---- RSF mol-view (nouveau head) ----
    print(f"\n>>> Training RSF (molecular view, cv_seed={cv_seed})")
    c_rsf, oof_rsf_seed, tst_rsf_seed = fit_rsf_mol(kf, X_tr_mol, X_te_mol,
                                                    random_state=RNG + 1000 + cv_seed)
    rsf_mol_oof_seeds.append(oof_rsf_seed)
    rsf_mol_tst_seeds.append(tst_rsf_seed)

# ----------------------------------------------------------------------
# CV-BAGGING ACROSS SEEDS
# ----------------------------------------------------------------------
oof_aft_cvbag = np.mean(aft_oof_seeds, axis=0)
tst_aft_cvbag = np.mean(aft_tst_seeds, axis=0)

oof_lgbm_full_cvbag = np.mean(lgb_full_oof_seeds, axis=0)
tst_lgbm_full_cvbag = np.mean(lgb_full_tst_seeds, axis=0)

oof_lgbm_mol_cvbag = np.mean(lgb_mol_oof_seeds, axis=0)
tst_lgbm_mol_cvbag = np.mean(lgb_mol_tst_seeds, axis=0)

oof_rsf_mol_cvbag = np.mean(rsf_mol_oof_seeds, axis=0)
tst_rsf_mol_cvbag = np.mean(rsf_mol_tst_seeds, axis=0)

c_aft_cvbag   = concordance_index_ipcw(y_sks, y_sks, oof_aft_cvbag)[0]
c_full_cvbag  = concordance_index_ipcw(y_sks, y_sks, oof_lgbm_full_cvbag)[0]
c_mol_cvbag   = concordance_index_ipcw(y_sks, y_sks, oof_lgbm_mol_cvbag)[0]
c_rsf_cvbag   = concordance_index_ipcw(y_sks, y_sks, oof_rsf_mol_cvbag)[0]

print(f"\nOOF IPCW-C -> XGB-AFT ensemble(4) + CV-bagging: {c_aft_cvbag:.4f}")
print(f"OOF IPCW-C -> LGBM full ensemble(3) + CV-bagging: {c_full_cvbag:.4f}")
print(f"OOF IPCW-C -> LGBM mol ensemble(2) + CV-bagging: {c_mol_cvbag:.4f}")
print(f"OOF IPCW-C -> RSF mol + CV-bagging: {c_rsf_cvbag:.4f}")

# ----------------------------------------------------------------------
# RANKS + BLENDS
#   - triple classique : (aft, lgb_full, lgb_mol)
#   - triple RSF       : (aft, lgb_full, rsf_mol)
# ----------------------------------------------------------------------
o_x    = to_rank(oof_aft_cvbag)
o_lf   = to_rank(oof_lgbm_full_cvbag)
o_lm   = to_rank(oof_lgbm_mol_cvbag)
o_rsf  = to_rank(oof_rsf_mol_cvbag)

t_x    = to_rank(tst_aft_cvbag)
t_lf   = to_rank(tst_lgbm_full_cvbag)
t_lm   = to_rank(tst_lgbm_mol_cvbag)
t_rsf  = to_rank(tst_rsf_mol_cvbag)

print("\n=== OOF C-index (ranks) par head ===")
print("OOF IPCW-C -> AFT only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_x)[0])
print("OOF IPCW-C -> LGBM full only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_lf)[0])
print("OOF IPCW-C -> LGBM mol only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_lm)[0])
print("OOF IPCW-C -> RSF mol only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_rsf)[0])

# Blends type G/H/I, classique et RSF
blend_specs = {
    # (label, which_mol_head)
    "G_lgbmol": ("lm", (0.57, 0.25, 0.18)),
    "H_lgbmol": ("lm", (0.50, 0.30, 0.20)),
    "I_lgbmol": ("lm", (0.58, 0.22, 0.20)),
    "G_rsf"   : ("rsf", (0.57, 0.25, 0.18)),
    "H_rsf"   : ("rsf", (0.50, 0.30, 0.20)),
    "I_rsf"   : ("rsf", (0.58, 0.22, 0.20)),
}

best_name = None
best_c    = -1
best_info = None

for name, (mol_tag, (w_x, w_lf, w_lm)) in blend_specs.items():
    w_sum = w_x + w_lf + w_lm
    if not np.isclose(w_sum, 1.0):
        w_x  /= w_sum
        w_lf /= w_sum
        w_lm /= w_sum

    if mol_tag == "lm":
        o_m = o_lm
        t_m = t_lm
    else:
        o_m = o_rsf
        t_m = t_rsf

    o_blend = w_x * o_x + w_lf * o_lf + w_lm * o_m
    c_blend = concordance_index_ipcw(y_sks, y_sks, o_blend)[0]
    print(f"OOF IPCW-C -> Blend {name} (mol={mol_tag}, w_x={w_x:.2f}, w_lf={w_lf:.2f}, w_m={w_lm:.2f}): {c_blend:.4f}")

    t_blend = w_x * t_x + w_lf * t_lf + w_lm * t_m
    out_name = f"submission_twin_v3_9_blend_{name}.csv"
    pd.DataFrame(
        {"ID": X_te.index, "risk_score": t_blend}
    ).set_index("ID").to_csv(out_name)
    print(f"✅ Wrote: {out_name}")

    if c_blend > best_c:
        best_c    = c_blend
        best_name = name
        best_info = (mol_tag, (w_x, w_lf, w_lm))

print(f"\nBest 3-head blend by OOF: {best_name} with C={best_c:.4f}, info={best_info}")

# also write raw heads (rankés)
pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_x}
).set_index("ID").to_csv("submission_twin_v3_9_head_aft.csv")

pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_lf}
).set_index("ID").to_csv("submission_twin_v3_9_head_lgb_full.csv")

pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_lm}
).set_index("ID").to_csv("submission_twin_v3_9_head_lgb_mol.csv")

pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_rsf}
).set_index("ID").to_csv("submission_twin_v3_9_head_rsf_mol.csv")

print("✅ Wrote heads + blends. Twin v3.9 finished.")
=== SHAPES RAW ===
clinical_train: (3323, 9)  clinical_test: (1193, 9)
molecular_train: (10935, 11)  molecular_test: (3089, 11)
target_train: (3323, 3)
Mol_agg shapes: (3026, 124) (1054, 111)
Features before clean: 157
Features final (before scaling): 178
Features final: 173
Mol-view features: 121 (vs 173 total)

=== CV seed 42 ===

>>> Training XGB-AFT (UNIFORM)

>>> AFT base_seed0 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed0: 0.7185

>>> AFT base_seed17 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed17: 0.7163

>>> AFT alt_seed0 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed0: 0.7166

>>> AFT alt_seed17 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed17: 0.7184
OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed=42]: 0.7195

>>> Training LGBM (full view, UNIFORM, cv_seed=42)
OOF IPCW-C -> LGBM(full_main, full-view): 0.7157
OOF IPCW-C -> LGBM(full_strong_reg, full-view): 0.7143
OOF IPCW-C -> LGBM(full_extra_reg, full-view): 0.7151
OOF IPCW-C -> LGBM ensemble(3, full view) [cv_seed=42]: 0.7156
LGBM full-view ensemble weights: {'full_main': np.float64(0.33363470128860817), 'full_strong_reg': np.float64(0.33300610863506125), 'full_extra_reg': np.float64(0.33335919007633075)}

>>> Training LGBM (molecular view, UNIFORM, cv_seed=42)
OOF IPCW-C -> LGBM(mol_main, mol-view): 0.6749
OOF IPCW-C -> LGBM(mol_strong, mol-view): 0.6754
OOF IPCW-C -> LGBM ensemble(2, mol view) [cv_seed=42]: 0.6753
LGBM mol-view ensemble weights: {'mol_main': np.float64(0.49980706251586515), 'mol_strong': np.float64(0.500192937484135)}

>>> Training RSF (molecular view, cv_seed=42)
OOF IPCW-C -> RSF(mol-view): 0.6735

=== CV seed 141 ===

>>> Training XGB-AFT (UNIFORM)

>>> AFT base_seed0 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed0: 0.7113

>>> AFT base_seed17 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed17: 0.7149

>>> AFT alt_seed0 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed0: 0.7121

>>> AFT alt_seed17 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed17: 0.7152
OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed=141]: 0.7156

>>> Training LGBM (full view, UNIFORM, cv_seed=141)
OOF IPCW-C -> LGBM(full_main, full-view): 0.7127
OOF IPCW-C -> LGBM(full_strong_reg, full-view): 0.7109
OOF IPCW-C -> LGBM(full_extra_reg, full-view): 0.7129
OOF IPCW-C -> LGBM ensemble(3, full view) [cv_seed=141]: 0.7127
LGBM full-view ensemble weights: {'full_main': np.float64(0.333593414928766), 'full_strong_reg': np.float64(0.33274416642148064), 'full_extra_reg': np.float64(0.3336624186497535)}

>>> Training LGBM (molecular view, UNIFORM, cv_seed=141)
OOF IPCW-C -> LGBM(mol_main, mol-view): 0.6742
OOF IPCW-C -> LGBM(mol_strong, mol-view): 0.6753
OOF IPCW-C -> LGBM ensemble(2, mol view) [cv_seed=141]: 0.6750
LGBM mol-view ensemble weights: {'mol_main': np.float64(0.49959773365080223), 'mol_strong': np.float64(0.5004022663491977)}

>>> Training RSF (molecular view, cv_seed=141)
OOF IPCW-C -> RSF(mol-view): 0.6748

=== CV seed 314 ===

>>> Training XGB-AFT (UNIFORM)

>>> AFT base_seed0 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed0: 0.7173

>>> AFT base_seed17 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed17: 0.7170

>>> AFT alt_seed0 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed0: 0.7167

>>> AFT alt_seed17 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed17: 0.7181
OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed=314]: 0.7193

>>> Training LGBM (full view, UNIFORM, cv_seed=314)
OOF IPCW-C -> LGBM(full_main, full-view): 0.7163
OOF IPCW-C -> LGBM(full_strong_reg, full-view): 0.7156
OOF IPCW-C -> LGBM(full_extra_reg, full-view): 0.7169
OOF IPCW-C -> LGBM ensemble(3, full view) [cv_seed=314]: 0.7168
LGBM full-view ensemble weights: {'full_main': np.float64(0.3333542616802529), 'full_strong_reg': np.float64(0.3330049454242911), 'full_extra_reg': np.float64(0.33364079289545606)}

>>> Training LGBM (molecular view, UNIFORM, cv_seed=314)
OOF IPCW-C -> LGBM(mol_main, mol-view): 0.6763
OOF IPCW-C -> LGBM(mol_strong, mol-view): 0.6765
OOF IPCW-C -> LGBM ensemble(2, mol view) [cv_seed=314]: 0.6766
LGBM mol-view ensemble weights: {'mol_main': np.float64(0.4999454483360147), 'mol_strong': np.float64(0.5000545516639853)}

>>> Training RSF (molecular view, cv_seed=314)
OOF IPCW-C -> RSF(mol-view): 0.6755

OOF IPCW-C -> XGB-AFT ensemble(4) + CV-bagging: 0.7210
OOF IPCW-C -> LGBM full ensemble(3) + CV-bagging: 0.7170
OOF IPCW-C -> LGBM mol ensemble(2) + CV-bagging: 0.6768
OOF IPCW-C -> RSF mol + CV-bagging: 0.6749

=== OOF C-index (ranks) par head ===
OOF IPCW-C -> AFT only (rank): 0.7210332355793426
OOF IPCW-C -> LGBM full only (rank): 0.716969372971452
OOF IPCW-C -> LGBM mol only (rank): 0.6768196383811953
OOF IPCW-C -> RSF mol only (rank): 0.6749011442347492
OOF IPCW-C -> Blend G_lgbmol (mol=lm, w_x=0.57, w_lf=0.25, w_m=0.18): 0.7225
✅ Wrote: submission_twin_v3_9_blend_G_lgbmol.csv
OOF IPCW-C -> Blend H_lgbmol (mol=lm, w_x=0.50, w_lf=0.30, w_m=0.20): 0.7219
✅ Wrote: submission_twin_v3_9_blend_H_lgbmol.csv
OOF IPCW-C -> Blend I_lgbmol (mol=lm, w_x=0.58, w_lf=0.22, w_m=0.20): 0.7222
✅ Wrote: submission_twin_v3_9_blend_I_lgbmol.csv
OOF IPCW-C -> Blend G_rsf (mol=rsf, w_x=0.57, w_lf=0.25, w_m=0.18): 0.7221
✅ Wrote: submission_twin_v3_9_blend_G_rsf.csv
OOF IPCW-C -> Blend H_rsf (mol=rsf, w_x=0.50, w_lf=0.30, w_m=0.20): 0.7216
✅ Wrote: submission_twin_v3_9_blend_H_rsf.csv
OOF IPCW-C -> Blend I_rsf (mol=rsf, w_x=0.58, w_lf=0.22, w_m=0.20): 0.7217
✅ Wrote: submission_twin_v3_9_blend_I_rsf.csv

Best 3-head blend by OOF: G_lgbmol with C=0.7225, info=('lm', (0.57, 0.25, 0.18))
✅ Wrote heads + blends. Twin v3.9 finished.
# ===================== QRT — Twin V3.10 (3-head robust) ===================== 
import os, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
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
    """Resolve path in ../data or ./."""
    for base in ("../data", "./"):
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

# ----------------------------------------------------------------------
# CYTO FEATURES (same base as v2.15)
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
# CLINICAL TRANSFORMS + SIMPLE RATIOS
# ----------------------------------------------------------------------
CLIN_CONT = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]

# missingness + logs + winsorization
for df in (clin_tr, clin_te):
    # missingness flags
    for c in CLIN_CONT:
        df[f"MISS_{c}"] = df[c].isna().astype(int)

    # logs (robust for skewed counts)
    for c in ["WBC", "ANC", "MONOCYTES", "PLT", "BM_BLAST"]:
        df[c] = np.log1p(df[c])

    # winsorisation
    for c in CLIN_CONT:
        df[c] = winsorize(df[c])

# quantiles pour flags de sévérité (basés sur train)
q_stats = {}
for c in CLIN_CONT:
    q_stats[c] = {
        "q10": clin_tr[c].quantile(0.10),
        "q90": clin_tr[c].quantile(0.90),
        "q75": clin_tr[c].quantile(0.75),
    }

for df in (clin_tr, clin_te):
    # ratios simples
    df["NEUT_RATIO"] = df["ANC"] / (df["WBC"] + 1e-6)
    df["MONO_RATIO"] = df["MONOCYTES"] / (df["WBC"] + 1e-6)
    df["BLAST_WBC_BURDEN"] = df["BM_BLAST"] * df["WBC"]

    # flags de sévérité
    df["HB_LOW"]        = (df["HB"] <= q_stats["HB"]["q10"]).astype(int)
    df["HB_HIGH"]       = (df["HB"] >= q_stats["HB"]["q90"]).astype(int)
    df["PLT_LOW"]       = (df["PLT"] <= q_stats["PLT"]["q10"]).astype(int)
    df["PLT_HIGH"]      = (df["PLT"] >= q_stats["PLT"]["q90"]).astype(int)
    df["WBC_HIGH"]      = (df["WBC"] >= q_stats["WBC"]["q90"]).astype(int)
    df["BM_BLAST_HIGH"] = (df["BM_BLAST"] >= q_stats["BM_BLAST"]["q75"]).astype(int)

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

# FLT3-ITD special flags
def build_special_mol_flags(df):
    by_id = df.groupby("ID")

    def any_flt3_itd(x):
        mask = (x["GENE"] == "FLT3") & x["EFFECT"].str.contains("ITD", case=False, na=False)
        return int(mask.any())

    def max_flt3_itd_vaf(x):
        mask = (x["GENE"] == "FLT3") & x["EFFECT"].str.contains("ITD", case=False, na=False)
        if mask.any():
            return float(x.loc[mask, "VAF"].max())
        else:
            return np.nan

    flt3_itd_any = by_id.apply(any_flt3_itd).rename("has__FLT3_ITD")
    flt3_itd_vaf = by_id.apply(max_flt3_itd_vaf).rename("FLT3_ITD_VAF_max")

    return pd.concat([flt3_itd_any, flt3_itd_vaf], axis=1)

mol_tr_special = build_special_mol_flags(mol_tr)
mol_te_special = build_special_mol_flags(mol_te)

def aggregate_mol(df, special=None):
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

    # thresholds (clonal / subclonal approx)
    n_ge_035 = g.apply(lambda x: (x["VAF"] >= 0.35).sum()).rename("n_mut_vaf_ge_035")
    n_ge_045 = g.apply(lambda x: (x["VAF"] >= 0.45).sum()).rename("n_mut_vaf_ge_045")
    n_lt_035 = g.apply(lambda x: (x["VAF"] < 0.35).sum()).rename("n_mut_vaf_lt_035")

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
            n_lt_035,
            eff,
            pres,
            path_ras,
            path_epi,
            path_spl,
            vaf_max,
        ],
        how="left",
    )

    # join FLT3-ITD special flags
    if special is not None:
        agg = agg.join(special, how="left")

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

    # simple subclonal burden
    agg["subclonal_burden"] = agg["VAF_sum"] - agg["VAF_max"]

    return agg

mol_tr_agg = aggregate_mol(mol_tr, special=mol_tr_special)
mol_te_agg = aggregate_mol(mol_te, special=mol_te_special)

print(f"Mol_agg shapes: {mol_tr_agg.shape} {mol_te_agg.shape}")

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
X_tr_raw.drop(columns=few_data_cols, inplace=True, errors="ignore")
X_te_raw.drop(columns=few_data_cols, inplace=True, errors="ignore")

# drop raw text cyto only
for df in (X_tr_raw, X_te_raw):
    if "CYTOGENETICS" in df.columns:
        df.drop(columns="CYTOGENETICS", inplace=True)

print(f"Features before clean: {len(all_cols)}")
print(f"Features final (before scaling): {X_tr_raw.shape[1]}")

# ----------------------------------------------------------------------
# IMPUTE + SCALE + VARIANCE FILTER + DROP HIGH CORR
# ----------------------------------------------------------------------
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
# AFT helpers
# ----------------------------------------------------------------------
y_time  = pd.Series(y_sks["time"],  index=X_tr.index).astype(float)
y_event = pd.Series(y_sks["event"], index=X_tr.index).astype(int)

def aft_bounds(t, e):
    lower = t.values.copy().astype(float)
    upper = t.values.copy().astype(float)
    upper[e.values == 0] = np.inf
    return lower, upper

aft_cfg_base = dict(
    objective="survival:aft",
    tree_method="hist",
    eta=0.045,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.7,
    reg_lambda=1.2,
    reg_alpha=0.0,
    aft_loss_distribution="normal",
    aft_loss_sigma=0.6,
)
aft_cfg_alt = dict(
    objective="survival:aft",
    tree_method="hist",
    eta=0.045,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.7,
    reg_lambda=1.2,
    reg_alpha=0.0,
    aft_loss_distribution="normal",
    aft_loss_sigma=0.8,
)

aft_cfgs = [
    ("base", aft_cfg_base, RNG),
    ("base", aft_cfg_base, RNG + 17),
    ("alt",  aft_cfg_alt,  RNG),
    ("alt",  aft_cfg_alt,  RNG + 17),
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
            num_boost_round=3500,
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

        teacher = CoxPHSurvivalAnalysis(alpha=1e-2, n_iter=800, tol=1e-9)
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

# full-view configs
lgb_full_cfg_main = dict(
    n_estimators=5500,
    learning_rate=0.03,
    num_leaves=80,
    min_data_in_leaf=60,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=1.5,
    reg_alpha=0.0,
    random_state=RNG,
    n_jobs=-1,
    verbose=-1,
)
lgb_full_cfg_strong = dict(
    n_estimators=5500,
    learning_rate=0.03,
    num_leaves=64,
    min_data_in_leaf=70,
    feature_fraction=0.65,
    bagging_fraction=0.75,
    bagging_freq=1,
    reg_lambda=2.0,
    reg_alpha=0.0,
    random_state=RNG + 7,
    n_jobs=-1,
    verbose=-1,
)
lgb_full_cfg_extra = dict(
    n_estimators=5500,
    learning_rate=0.03,
    num_leaves=48,
    min_data_in_leaf=80,
    feature_fraction=0.60,
    bagging_fraction=0.75,
    bagging_freq=1,
    reg_lambda=2.5,
    reg_alpha=0.0,
    random_state=RNG + 21,
    n_jobs=-1,
    verbose=-1,
)

lgb_full_configs = [
    ("full_main",       lgb_full_cfg_main),
    ("full_strong_reg", lgb_full_cfg_strong),
    ("full_extra_reg",  lgb_full_cfg_extra),
]

# mol-view configs (un peu plus petits)
lgb_mol_cfg_main = dict(
    n_estimators=4500,
    learning_rate=0.03,
    num_leaves=64,
    min_data_in_leaf=40,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=1.5,
    reg_alpha=0.0,
    random_state=RNG + 101,
    n_jobs=-1,
    verbose=-1,
)
lgb_mol_cfg_strong = dict(
    n_estimators=4500,
    learning_rate=0.03,
    num_leaves=48,
    min_data_in_leaf=50,
    feature_fraction=0.75,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=2.0,
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
    print("\n>>> Training XGB-AFT (UNIFORM)")
    for name, cfg, s in aft_cfgs:
        print(f"\n>>> AFT {name}_seed{s-RNG} [UNIFORM]")
        c, oof_c, tst_c = fit_aft_cfg(cfg, s, kf)
        print(f"OOF IPCW-C -> AFT({name}, UNIFORM) seed{s-RNG}: {c:.4f}")
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

    print(f"\n>>> Training LGBM (full view, UNIFORM, cv_seed={cv_seed})")
    for tag, cfg in lgb_full_configs:
        c, oof_c, tst_c = fit_lgbm_cfg_view(cfg, f"{tag}, full-view", kf, X_tr, X_te)
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

    print(f"\n>>> Training LGBM (molecular view, UNIFORM, cv_seed={cv_seed})")
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
# RANKS + 3-HEAD BLENDS + CSV SUBMISSIONS
# ----------------------------------------------------------------------
o_x  = to_rank(oof_aft_cvbag)
o_lf = to_rank(oof_lgbm_full_cvbag)
o_lm = to_rank(oof_lgbm_mol_cvbag)

t_x  = to_rank(tst_aft_cvbag)
t_lf = to_rank(tst_lgbm_full_cvbag)
t_lm = to_rank(tst_lgbm_mol_cvbag)

print("=== OOF C-index (ranks) par head ===")
print("OOF IPCW-C -> AFT only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_x)[0])
print("OOF IPCW-C -> LGBM full only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_lf)[0])
print("OOF IPCW-C -> LGBM mol only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_lm)[0])

blend_configs = {
    "G": (0.57, 0.25, 0.18),  # w_x, w_l_full, w_l_mol
    "H": (0.50, 0.30, 0.20),
    "I": (0.58, 0.22, 0.20),
}

version_tag = "v3_10"

best_name = None
best_c = -1.0

for name, (w_x, w_lf, w_lm) in blend_configs.items():
    w_sum = w_x + w_lf + w_lm
    if not np.isclose(w_sum, 1.0):
        print(f"[WARN] Blend {name}: weights do not sum to 1 (sum={w_sum:.4f}), renormalising.")
        w_x  /= w_sum
        w_lf /= w_sum
        w_lm /= w_sum

    o_blend = w_x * o_x + w_lf * o_lf + w_lm * o_lm
    c_blend = concordance_index_ipcw(y_sks, y_sks, o_blend)[0]
    print(f"OOF IPCW-C -> Blend {name} (w_x={w_x:.2f}, w_lf={w_lf:.2f}, w_lm={w_lm:.2f}): {c_blend:.4f}")

    t_blend = w_x * t_x + w_lf * t_lf + w_lm * t_lm
    out_name = f"submission_twin_{version_tag}_blend_{name}_wx{w_x:.2f}_wlf{w_lf:.2f}_wlm{w_lm:.2f}.csv"
    pd.DataFrame(
        {"ID": X_te.index, "risk_score": t_blend}
    ).set_index("ID").to_csv(out_name)
    print(f"✅ Wrote: {out_name}")

    if c_blend > best_c:
        best_c = c_blend
        best_name = name

print(f"\nBest 3-head blend by OOF: {best_name} with C={best_c:.4f}")

# Also write raw heads
out_head_aft = f"submission_twin_{version_tag}_head_aft.csv"
out_head_lf  = f"submission_twin_{version_tag}_head_lgbm_full.csv"
out_head_lm  = f"submission_twin_{version_tag}_head_lgbm_mol.csv"

pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_x}
).set_index("ID").to_csv(out_head_aft)

pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_lf}
).set_index("ID").to_csv(out_head_lf)

pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_lm}
).set_index("ID").to_csv(out_head_lm)

print(f"✅ Wrote base heads: {out_head_aft}, {out_head_lf}, {out_head_lm}")
print(f"✅ Twin {version_tag} finished.")
=== SHAPES RAW ===
clinical_train: (3323, 9)  clinical_test: (1193, 9)
molecular_train: (10935, 11)  molecular_test: (3089, 11)
target_train: (3323, 3)
Mol_agg shapes: (3026, 125) (1054, 112)
Features before clean: 167
Features final (before scaling): 188
Features final: 184
Mol-view features: 123 (vs 184 total)

=== CV seed 42 ===

>>> Training XGB-AFT (UNIFORM)

>>> AFT base_seed0 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed0: 0.7197

>>> AFT base_seed17 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed17: 0.7164

>>> AFT alt_seed0 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed0: 0.7190

>>> AFT alt_seed17 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed17: 0.7157
OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed=42]: 0.7202

>>> Training LGBM (full view, UNIFORM, cv_seed=42)
OOF IPCW-C -> LGBM(full_main, full-view): 0.7167
OOF IPCW-C -> LGBM(full_strong_reg, full-view): 0.7153
OOF IPCW-C -> LGBM(full_extra_reg, full-view): 0.7167
OOF IPCW-C -> LGBM ensemble(3, full view) [cv_seed=42]: 0.7166
LGBM full-view ensemble weights: {'full_main': np.float64(0.33354094865796374), 'full_strong_reg': np.float64(0.3329255841895851), 'full_extra_reg': np.float64(0.33353346715245097)}

>>> Training LGBM (molecular view, UNIFORM, cv_seed=42)
OOF IPCW-C -> LGBM(mol_main, mol-view): 0.6749
OOF IPCW-C -> LGBM(mol_strong, mol-view): 0.6754
OOF IPCW-C -> LGBM ensemble(2, mol view) [cv_seed=42]: 0.6752
LGBM mol-view ensemble weights: {'mol_main': np.float64(0.4998102699428237), 'mol_strong': np.float64(0.5001897300571764)}

=== CV seed 141 ===

>>> Training XGB-AFT (UNIFORM)

>>> AFT base_seed0 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed0: 0.7135

>>> AFT base_seed17 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed17: 0.7122

>>> AFT alt_seed0 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed0: 0.7142

>>> AFT alt_seed17 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed17: 0.7133
OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed=141]: 0.7154

>>> Training LGBM (full view, UNIFORM, cv_seed=141)
OOF IPCW-C -> LGBM(full_main, full-view): 0.7130
OOF IPCW-C -> LGBM(full_strong_reg, full-view): 0.7116
OOF IPCW-C -> LGBM(full_extra_reg, full-view): 0.7120
OOF IPCW-C -> LGBM ensemble(3, full view) [cv_seed=141]: 0.7128
LGBM full-view ensemble weights: {'full_main': np.float64(0.33372550084763664), 'full_strong_reg': np.float64(0.3330624359135559), 'full_extra_reg': np.float64(0.3332120632388075)}

>>> Training LGBM (molecular view, UNIFORM, cv_seed=141)
OOF IPCW-C -> LGBM(mol_main, mol-view): 0.6737
OOF IPCW-C -> LGBM(mol_strong, mol-view): 0.6736
OOF IPCW-C -> LGBM ensemble(2, mol view) [cv_seed=141]: 0.6739
LGBM mol-view ensemble weights: {'mol_main': np.float64(0.5000413121573294), 'mol_strong': np.float64(0.4999586878426705)}

=== CV seed 314 ===

>>> Training XGB-AFT (UNIFORM)

>>> AFT base_seed0 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed0: 0.7157

>>> AFT base_seed17 [UNIFORM]
OOF IPCW-C -> AFT(base, UNIFORM) seed17: 0.7176

>>> AFT alt_seed0 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed0: 0.7158

>>> AFT alt_seed17 [UNIFORM]
OOF IPCW-C -> AFT(alt, UNIFORM) seed17: 0.7165
OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed=314]: 0.7184

>>> Training LGBM (full view, UNIFORM, cv_seed=314)
OOF IPCW-C -> LGBM(full_main, full-view): 0.7158
OOF IPCW-C -> LGBM(full_strong_reg, full-view): 0.7152
OOF IPCW-C -> LGBM(full_extra_reg, full-view): 0.7154
OOF IPCW-C -> LGBM ensemble(3, full view) [cv_seed=314]: 0.7161
LGBM full-view ensemble weights: {'full_main': np.float64(0.3335174713245761), 'full_strong_reg': np.float64(0.3331953259830076), 'full_extra_reg': np.float64(0.3332872026924163)}

>>> Training LGBM (molecular view, UNIFORM, cv_seed=314)
OOF IPCW-C -> LGBM(mol_main, mol-view): 0.6765
OOF IPCW-C -> LGBM(mol_strong, mol-view): 0.6763
OOF IPCW-C -> LGBM ensemble(2, mol view) [cv_seed=314]: 0.6766
LGBM mol-view ensemble weights: {'mol_main': np.float64(0.5000987836350108), 'mol_strong': np.float64(0.49990121636498924)}

OOF IPCW-C -> XGB-AFT ensemble(4) + CV-bagging: 0.7210
OOF IPCW-C -> LGBM full ensemble(3) + CV-bagging: 0.7174
OOF IPCW-C -> LGBM mol-view ensemble(2) + CV-bagging: 0.6765
=== OOF C-index (ranks) par head ===
OOF IPCW-C -> AFT only (rank): 0.7210184226234965
OOF IPCW-C -> LGBM full only (rank): 0.7173528784992794
OOF IPCW-C -> LGBM mol only (rank): 0.6764772179494716
OOF IPCW-C -> Blend G (w_x=0.57, w_lf=0.25, w_lm=0.18): 0.7227
✅ Wrote: submission_twin_v3_10_blend_G_wx0.57_wlf0.25_wlm0.18.csv
OOF IPCW-C -> Blend H (w_x=0.50, w_lf=0.30, w_lm=0.20): 0.7222
✅ Wrote: submission_twin_v3_10_blend_H_wx0.50_wlf0.30_wlm0.20.csv
OOF IPCW-C -> Blend I (w_x=0.58, w_lf=0.22, w_lm=0.20): 0.7224
✅ Wrote: submission_twin_v3_10_blend_I_wx0.58_wlf0.22_wlm0.20.csv

Best 3-head blend by OOF: G with C=0.7227
✅ Wrote base heads: submission_twin_v3_10_head_aft.csv, submission_twin_v3_10_head_lgbm_full.csv, submission_twin_v3_10_head_lgbm_mol.csv
✅ Twin v3_10 finished.
# ===================== QRT — Twin V5 (G-like + univariate mol feature filtering) =====================
import os, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis

import xgboost as xgb
import lightgbm as lgb

# ----------------------------------------------------------------------
# GLOBAL PARAMS
# ----------------------------------------------------------------------
RNG = 42
N_FOLDS = 5
CV_SEEDS = [42, 141, 314]

TOP_MOL_KEEP = 70  # nombre de features moléculaires à garder (à ajuster si tu veux)

# ----------------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------------
def rp(path):
    """
    Resolve path either in ../data or current dir.
    """
    for base in ("../data", "./"):
        p = os.path.join(base, path)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(path)

def to_rank(x):
    """
    Convert scores to [0,1] ranks (monotonic but scale-free).
    """
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

    # logs
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
TOP_N_GENES   = 70   # même choix que v2.15
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
print(f"Mol_agg shapes: {mol_tr_agg.shape} {mol_te_agg.shape}")

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

    # score ordinal ELN-like (0=fav,1=int,2=adv)
    X["eln_like_score"] = eln_fav * 0 + eln_int * 1 + eln_adv * 2

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
# IMPUTE + SCALE + VARIANCE FILTER + DROP HIGH CORR
# ----------------------------------------------------------------------
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
drop_corr = sorted(set(corr.columns[j] for j in hi_j))
X_tr.drop(columns=drop_corr, inplace=True, errors="ignore")
X_te = X_te.reindex(columns=X_tr.columns).fillna(0.0)

# clip for stability
X_tr = X_tr.clip(-6, 6).astype("float32")
X_te = X_te.clip(-6, 6).astype("float32")

print(f"Features final (after corr filter): {X_tr.shape[1]}")

# ----------------------------------------------------------------------
# NEW STEP v5 : univariate Cox feature selection on mol features
# ----------------------------------------------------------------------
mol_features_all = [c for c in X_tr.columns if c in mol_tr_agg.columns]
print(f"Mol features raw in X_tr: {len(mol_features_all)}")

univ_scores = []
cox_uni = CoxPHSurvivalAnalysis(alpha=1e-3, n_iter=200, tol=1e-7)

for col in mol_features_all:
    try:
        # single-feature Cox, data already standardisé
        cox_uni.fit(X_tr[[col]], y_sks)
        coef = float(cox_uni.coef_[0])
        score = abs(coef)
    except Exception:
        score = 0.0
    univ_scores.append((col, score))

univ_df = pd.DataFrame(univ_scores, columns=["feature", "score"])
univ_df.sort_values("score", ascending=False, inplace=True)

top_mol_cols = list(univ_df.head(TOP_MOL_KEEP)["feature"])
drop_mol_cols = sorted(set(mol_features_all) - set(top_mol_cols))

print(f"Keeping top {len(top_mol_cols)} mol features, dropping {len(drop_mol_cols)}.")
if drop_mol_cols:
    X_tr.drop(columns=drop_mol_cols, inplace=True)
    X_te.drop(columns=drop_mol_cols, inplace=True)

print(f"Features final after mol-selection: {X_tr.shape[1]}")

# ----------------------------------------------------------------------
# MOLECULAR VIEW (subset of features) — after selection
# ----------------------------------------------------------------------
mol_view_cols = [c for c in X_tr.columns if c in mol_tr_agg.columns]
X_tr_mol = X_tr[mol_view_cols].copy()
X_te_mol = X_te[mol_view_cols].copy()
print(f"Mol-view features: {len(mol_view_cols)} (vs {X_tr.shape[1]} total)")

# ----------------------------------------------------------------------
# AFT helpers
# ----------------------------------------------------------------------
y_time  = pd.Series(y_sks["time"],  index=X_tr.index).astype(float)
y_event = pd.Series(y_sks["event"], index=X_tr.index).astype(int)

def aft_bounds(t, e):
    lower = t.values.copy().astype(float)
    upper = t.values.copy().astype(float)
    upper[e.values == 0] = np.inf
    return lower, upper

aft_cfg_base = dict(
    objective="survival:aft",
    tree_method="hist",
    eta=0.045,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.7,
    reg_lambda=1.2,
    reg_alpha=0.0,
    aft_loss_distribution="normal",
    aft_loss_sigma=0.6,
)
aft_cfg_alt = dict(
    objective="survival:aft",
    tree_method="hist",
    eta=0.045,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.7,
    reg_lambda=1.2,
    reg_alpha=0.0,
    aft_loss_distribution="normal",
    aft_loss_sigma=0.8,
)

aft_cfgs = [
    ("base", aft_cfg_base, RNG),
    ("base", aft_cfg_base, RNG + 17),
    ("alt",  aft_cfg_alt,  RNG),
    ("alt",  aft_cfg_alt,  RNG + 17),
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
            num_boost_round=3500,
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
# LGBM helpers (full & mol views) — identiques à v2.15
# ----------------------------------------------------------------------
def fit_lgbm_cfg_view(lgb_params, tag, kf, X_tr_view, X_te_view):
    oof = np.zeros(len(X_tr_view))
    tst = np.zeros(len(X_te_view))

    for tr_idx, va_idx in kf.split(X_tr_view):
        Xt, Xv = X_tr_view.iloc[tr_idx], X_tr_view.iloc[va_idx]
        yt, yv = y_sks[tr_idx], y_sks[va_idx]

        teacher = CoxPHSurvivalAnalysis(alpha=1e-2, n_iter=800, tol=1e-9)
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

# full-view configs
lgb_full_cfg_main = dict(
    n_estimators=5500,
    learning_rate=0.03,
    num_leaves=80,
    min_data_in_leaf=60,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=1.5,
    reg_alpha=0.0,
    random_state=RNG,
    n_jobs=-1,
    verbose=-1,
)
lgb_full_cfg_strong = dict(
    n_estimators=5500,
    learning_rate=0.03,
    num_leaves=64,
    min_data_in_leaf=70,
    feature_fraction=0.65,
    bagging_fraction=0.75,
    bagging_freq=1,
    reg_lambda=2.0,
    reg_alpha=0.0,
    random_state=RNG + 7,
    n_jobs=-1,
    verbose=-1,
)
lgb_full_cfg_extra = dict(
    n_estimators=5500,
    learning_rate=0.03,
    num_leaves=48,
    min_data_in_leaf=80,
    feature_fraction=0.60,
    bagging_fraction=0.75,
    bagging_freq=1,
    reg_lambda=2.5,
    reg_alpha=0.0,
    random_state=RNG + 21,
    n_jobs=-1,
    verbose=-1,
)

lgb_full_configs = [
    ("full_main",       lgb_full_cfg_main),
    ("full_strong_reg", lgb_full_cfg_strong),
    ("full_extra_reg",  lgb_full_cfg_extra),
]

# mol-view configs
lgb_mol_cfg_main = dict(
    n_estimators=4500,
    learning_rate=0.03,
    num_leaves=64,
    min_data_in_leaf=40,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=1.5,
    reg_alpha=0.0,
    random_state=RNG + 101,
    n_jobs=-1,
    verbose=-1,
)
lgb_mol_cfg_strong = dict(
    n_estimators=4500,
    learning_rate=0.03,
    num_leaves=48,
    min_data_in_leaf=50,
    feature_fraction=0.75,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=2.0,
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
# RANKS + BLEND (G-style 3-head)
# ----------------------------------------------------------------------
o_x  = to_rank(oof_aft_cvbag)
o_lf = to_rank(oof_lgbm_full_cvbag)
o_lm = to_rank(oof_lgbm_mol_cvbag)

t_x  = to_rank(tst_aft_cvbag)
t_lf = to_rank(tst_lgbm_full_cvbag)
t_lm = to_rank(tst_lgbm_mol_cvbag)

print("\n=== OOF C-index (ranks) par head ===")
print("OOF IPCW-C -> AFT only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_x)[0])
print("OOF IPCW-C -> LGBM full only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_lf)[0])
print("OOF IPCW-C -> LGBM mol-view only (rank):",
      concordance_index_ipcw(y_sks, y_sks, o_lm)[0])

# Blend G3 (3-head, G-like)
w_x, w_lf, w_lm = 0.57, 0.25, 0.18
w_sum = w_x + w_lf + w_lm
if not np.isclose(w_sum, 1.0):
    w_x  /= w_sum
    w_lf /= w_sum
    w_lm /= w_sum
    print(f"[WARN] Blend G3: weights renormalised to sum=1: "
          f"w_x={w_x:.3f}, w_lf={w_lf:.3f}, w_lm={w_lm:.3f}")

o_blend = w_x * o_x + w_lf * o_lf + w_lm * o_lm
c_blend = concordance_index_ipcw(y_sks, y_sks, o_blend)[0]
print(f"OOF IPCW-C -> Blend G3 (w_x={w_x:.2f}, w_lf={w_lf:.2f}, w_lm={w_lm:.2f}): {c_blend:.4f}")

t_blend = w_x * t_x + w_lf * t_lf + w_lm * t_lm
out_name = f"submission_twin_v5_blend_G3_wx{w_x:.2f}_wlf{w_lf:.2f}_wlm{w_lm:.2f}.csv"
pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_blend}
).set_index("ID").to_csv(out_name)
print(f"✅ Wrote: {out_name}")

# ----------------------------------------------------------------------
# Also write raw model heads (pour debug / stacking)
# ----------------------------------------------------------------------
pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_x}
).set_index("ID").to_csv("submission_twin_v5_head_aft_ensemble_cvbag.csv")

pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_lf}
).set_index("ID").to_csv("submission_twin_v5_head_lgbm_full_ens3_cvbag_rank.csv")

pd.DataFrame(
    {"ID": X_te.index, "risk_score": t_lm}
).set_index("ID").to_csv("submission_twin_v5_head_lgbm_mol_ens2_cvbag_rank.csv")

print("✅ Wrote base heads + G3 blend (v5).")
=== SHAPES RAW ===
clinical_train: (3323, 9)  clinical_test: (1193, 9)
molecular_train: (10935, 11)  molecular_test: (3089, 11)
target_train: (3323, 3)
Mol_agg shapes: (3026, 121) (1054, 108)
Features before clean: 155
Features final (before scaling): 176
Features final (after corr filter): 172
Mol features raw in X_tr: 119
Keeping top 70 mol features, dropping 49.
Features final after mol-selection: 123
Mol-view features: 70 (vs 123 total)

=== CV seed 42 ===

>>> Training AFT base_seed0 (cv_seed=42)
OOF IPCW-C -> AFT(base) seed0: 0.7157

>>> Training AFT base_seed17 (cv_seed=42)
OOF IPCW-C -> AFT(base) seed17: 0.7172

>>> Training AFT alt_seed0 (cv_seed=42)
OOF IPCW-C -> AFT(alt) seed0: 0.7159

>>> Training AFT alt_seed17 (cv_seed=42)
OOF IPCW-C -> AFT(alt) seed17: 0.7158
OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed=42]: 0.7182

>>> Training LGBM (full view, cv_seed=42)
OOF IPCW-C -> LGBM(full_main): 0.7149
OOF IPCW-C -> LGBM(full_strong_reg): 0.7140
OOF IPCW-C -> LGBM(full_extra_reg): 0.7141
OOF IPCW-C -> LGBM ensemble(3, full view) [cv_seed=42]: 0.7148
LGBM full-view ensemble weights: {'full_main': np.float64(0.33359822737744566), 'full_strong_reg': np.float64(0.3331738580318408), 'full_extra_reg': np.float64(0.33322791459071344)}

>>> Training LGBM (molecular view, cv_seed=42)
OOF IPCW-C -> LGBM(mol_main, mol-view): 0.6734
OOF IPCW-C -> LGBM(mol_strong, mol-view): 0.6732
OOF IPCW-C -> LGBM ensemble(2, mol view) [cv_seed=42]: 0.6733
LGBM mol-view ensemble weights: {'mol_main': np.float64(0.5000821961257453), 'mol_strong': np.float64(0.4999178038742546)}

=== CV seed 141 ===

>>> Training AFT base_seed0 (cv_seed=141)
OOF IPCW-C -> AFT(base) seed0: 0.7141

>>> Training AFT base_seed17 (cv_seed=141)
OOF IPCW-C -> AFT(base) seed17: 0.7140

>>> Training AFT alt_seed0 (cv_seed=141)
OOF IPCW-C -> AFT(alt) seed0: 0.7142

>>> Training AFT alt_seed17 (cv_seed=141)
OOF IPCW-C -> AFT(alt) seed17: 0.7157
OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed=141]: 0.7164

>>> Training LGBM (full view, cv_seed=141)
OOF IPCW-C -> LGBM(full_main): 0.7111
OOF IPCW-C -> LGBM(full_strong_reg): 0.7107
OOF IPCW-C -> LGBM(full_extra_reg): 0.7118
OOF IPCW-C -> LGBM ensemble(3, full view) [cv_seed=141]: 0.7116
LGBM full-view ensemble weights: {'full_main': np.float64(0.3332796437369746), 'full_strong_reg': np.float64(0.3331182467107961), 'full_extra_reg': np.float64(0.33360210955222935)}

>>> Training LGBM (molecular view, cv_seed=141)
OOF IPCW-C -> LGBM(mol_main, mol-view): 0.6707
OOF IPCW-C -> LGBM(mol_strong, mol-view): 0.6716
OOF IPCW-C -> LGBM ensemble(2, mol view) [cv_seed=141]: 0.6713
LGBM mol-view ensemble weights: {'mol_main': np.float64(0.4996622741440078), 'mol_strong': np.float64(0.5003377258559921)}

=== CV seed 314 ===

>>> Training AFT base_seed0 (cv_seed=314)
OOF IPCW-C -> AFT(base) seed0: 0.7160

>>> Training AFT base_seed17 (cv_seed=314)
OOF IPCW-C -> AFT(base) seed17: 0.7173

>>> Training AFT alt_seed0 (cv_seed=314)
OOF IPCW-C -> AFT(alt) seed0: 0.7162

>>> Training AFT alt_seed17 (cv_seed=314)
OOF IPCW-C -> AFT(alt) seed17: 0.7172
OOF IPCW-C -> XGB-AFT ensemble(4) [cv_seed=314]: 0.7187

>>> Training LGBM (full view, cv_seed=314)
OOF IPCW-C -> LGBM(full_main): 0.7143
OOF IPCW-C -> LGBM(full_strong_reg): 0.7141
OOF IPCW-C -> LGBM(full_extra_reg): 0.7131
OOF IPCW-C -> LGBM ensemble(3, full view) [cv_seed=314]: 0.7144
LGBM full-view ensemble weights: {'full_main': np.float64(0.3335498260513309), 'full_strong_reg': np.float64(0.3334712640213044), 'full_extra_reg': np.float64(0.3329789099273647)}

>>> Training LGBM (molecular view, cv_seed=314)
OOF IPCW-C -> LGBM(mol_main, mol-view): 0.6734
OOF IPCW-C -> LGBM(mol_strong, mol-view): 0.6737
OOF IPCW-C -> LGBM ensemble(2, mol view) [cv_seed=314]: 0.6738
LGBM mol-view ensemble weights: {'mol_main': np.float64(0.49989373459544095), 'mol_strong': np.float64(0.500106265404559)}

OOF IPCW-C -> XGB-AFT ensemble(4) + CV-bagging: 0.7204
OOF IPCW-C -> LGBM full ensemble(3) + CV-bagging: 0.7155
OOF IPCW-C -> LGBM mol-view ensemble(2) + CV-bagging: 0.6736

=== OOF C-index (ranks) par head ===
OOF IPCW-C -> AFT only (rank): 0.7204212550786534
OOF IPCW-C -> LGBM full only (rank): 0.7154800215513177
OOF IPCW-C -> LGBM mol-view only (rank): 0.6736042107327286
OOF IPCW-C -> Blend G3 (w_x=0.57, w_lf=0.25, w_lm=0.18): 0.7219
✅ Wrote: submission_twin_v5_blend_G3_wx0.57_wlf0.25_wlm0.18.csv
✅ Wrote base heads + G3 blend (v5).
