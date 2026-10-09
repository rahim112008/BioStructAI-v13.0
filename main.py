"""
BioStruct AI v14.0 - Plateforme intégrée de bioinformatique structurale
Multi-LLM : OpenAI, Groq, Gemini, Mistral, Cerebras, OpenRouter, Ollama, HuggingFace
37+ modules - Fichier unique
"""

import streamlit as st
import py3Dmol
from stmol import showmol
import pandas as pd
import numpy as np
import requests
from io import StringIO
import tempfile, os, json, hashlib, importlib, inspect, io
from datetime import datetime
from collections import Counter
from abc import ABC, abstractmethod

from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_dihedral

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (roc_curve, auc, roc_auc_score, accuracy_score,
                              precision_score, recall_score, f1_score)
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, ttest_ind
from statsmodels.stats.multitest import multipletests

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from pandadock.protein import Protein
    from pandadock.ligand import Ligand
    from pandadock.scoring import CompositeScoringFunction
    PANDADOCK_AVAILABLE = True
except ImportError:
    PANDADOCK_AVAILABLE = False

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================
# SECTION 1 : CLIENTS
# ============================================================

@st.cache_resource
def get_supabase_client():
    if not SUPABASE_AVAILABLE: return None
    try:
        return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])
    except Exception:
        return None

@st.cache_resource
def get_openai_client():
    if not OPENAI_AVAILABLE: return None
    try:
        return OpenAI(api_key=st.secrets["openai"]["api_key"])
    except Exception:
        return None


# ============================================================
# SECTION 2 : PERSISTANCE
# ============================================================

def get_or_create_user(email):
    supabase = get_supabase_client()
    if not supabase: return {"id": 1, "email": email, "offline": True}
    try:
        res = supabase.table("users").select("*").eq("email", email).execute()
        if res.data: return res.data[0]
        return supabase.table("users").insert({"email": email}).execute().data[0]
    except Exception:
        return None

def save_analysis(uid, atype, inputs, results, pid=None, capsule=None):
    supabase = get_supabase_client()
    if not supabase: return None
    try:
        return supabase.table("analyses").insert({
            "user_id": uid, "analysis_type": atype,
            "input_data": inputs, "results": results,
            "project_id": pid, "reproducibility_capsule": capsule,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception:
        return None

def load_analyses(uid):
    supabase = get_supabase_client()
    if not supabase: return []
    try:
        return supabase.table("analyses").select("*").eq("user_id", uid).order("created_at", desc=True).execute().data
    except Exception:
        return []

def generate_capsule(atype, inputs, params, version="14.0.0"):
    chk = json.dumps(inputs, sort_keys=True, default=str)
    return {"version": version, "analysis_type": atype,
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs, "parameters": params,
            "checksum": hashlib.sha256(chk.encode()).hexdigest()[:16]}


# ============================================================
# SECTION 3 : CONSTANTES
# ============================================================

AA_PROPERTIES = {
    'ALA': ( 1.8,  88.6, 0, 0, 0, 0.36), 'ARG': (-4.5, 173.4, 1, 1, 0, 0.53),
    'ASN': (-3.5, 114.1, 0, 1, 0, 0.46), 'ASP': (-3.5, 111.1, -1, 1, 0, 0.51),
    'CYS': ( 2.5, 108.5, 0, 0, 0, 0.35), 'GLN': (-3.5, 143.8, 0, 1, 0, 0.49),
    'GLU': (-3.5, 138.4, -1, 1, 0, 0.50), 'GLY': (-0.4,  60.1, 0, 0, 0, 0.54),
    'HIS': (-3.2, 153.2, 1, 1, 1, 0.32), 'ILE': ( 4.5, 166.7, 0, 0, 0, 0.46),
    'LEU': ( 3.8, 166.7, 0, 0, 0, 0.37), 'LYS': (-3.9, 168.6, 1, 1, 0, 0.47),
    'MET': ( 1.9, 162.9, 0, 0, 0, 0.30), 'PHE': ( 2.8, 189.9, 0, 0, 1, 0.31),
    'PRO': (-1.6, 112.7, 0, 0, 0, 0.51), 'SER': (-0.8,  89.0, 0, 1, 0, 0.51),
    'THR': (-0.7, 116.1, 0, 1, 0, 0.44), 'TRP': (-0.9, 227.8, 0, 0, 1, 0.31),
    'TYR': (-1.3, 193.6, 0, 1, 1, 0.42), 'VAL': ( 4.2, 140.0, 0, 0, 0, 0.39),
}

POPULATION_VARIANTS_DB = {
    "SLC24A5": {"rs1426654": {"A": {"Europe": 0.95, "Afrique": 0.05, "Asie": 0.60},
                              "G": {"Europe": 0.05, "Afrique": 0.95, "Asie": 0.40}}},
    "TYR": {"rs1042602": {"C": {"Europe": 0.60, "Afrique": 0.95, "Asie": 0.85},
                          "A": {"Europe": 0.40, "Afrique": 0.05, "Asie": 0.15}}},
    "LCT": {"rs4988235": {"T": {"Europe": 0.75, "Afrique": 0.10, "Asie": 0.05},
                          "C": {"Europe": 0.25, "Afrique": 0.90, "Asie": 0.95}}},
    "HBB": {"rs334": {"A": {"Europe": 0.00, "Afrique": 0.15, "Asie": 0.02},
                      "T": {"Europe": 1.00, "Afrique": 0.85, "Asie": 0.98}}},
    "DARC": {"rs2814778": {"T": {"Europe": 0.00, "Afrique": 0.95, "Asie": 0.05},
                           "C": {"Europe": 1.00, "Afrique": 0.05, "Asie": 0.95}}},
    "APOL1": {"rs73885319": {"A": {"Europe": 0.00, "Afrique": 0.35, "Asie": 0.00},
                             "G": {"Europe": 1.00, "Afrique": 0.65, "Asie": 1.00}}},
    "F13B": {"rs6003": {"A": {"Europe": 0.30, "Afrique": 0.70, "Asie": 0.50},
                        "G": {"Europe": 0.70, "Afrique": 0.30, "Asie": 0.50}}},
    "CYP2D6": {"rs3892097": {"A": {"Europe": 0.20, "Afrique": 0.05, "Asie": 0.01},
                             "G": {"Europe": 0.80, "Afrique": 0.95, "Asie": 0.99}}},
}

UALCAN_CANCERS = {
    "BRCA": "Breast", "LUAD": "Lung adeno", "LUSC": "Lung squamous",
    "COAD": "Colon", "GBM": "Glioblastoma", "HNSC": "Head & neck",
    "KIRC": "Kidney", "LIHC": "Liver", "OV": "Ovarian",
    "PAAD": "Pancreatic", "PRAD": "Prostate", "STAD": "Stomach",
    "THCA": "Thyroid", "UCEC": "Endometrial"
}


# ============================================================
# SECTION 4 : PDB & VALIDATION
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pdb_structure(pdb_id):
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=30)
    if r.status_code == 200: return r.text
    raise ValueError(f"Structure {pdb_id} introuvable.")

@st.cache_data(ttl=3600, show_spinner=False)
def parse_pdb_structure(txt):
    return PDBParser(PERMISSIVE=1).get_structure("s", StringIO(txt))

def get_structure_info(structure):
    chains = list(structure.get_chains())
    return {"Numéro PDB": structure.header.get("idcode", "N/A"),
            "Nom": structure.header.get("name", "N/A"),
            "Chaînes": [c.id for c in chains],
            "Résidus": sum(1 for _ in structure.get_residues()),
            "Atomes": sum(1 for _ in structure.get_atoms())}

def calculate_phi_psi(structure):
    results = []
    for chain in structure.get_chains():
        residues = [r for r in chain.get_residues() if r.id[0] == " "]
        for i in range(1, len(residues) - 1):
            try:
                prev_c = residues[i-1]["C"].get_vector()
                n = residues[i]["N"].get_vector()
                ca = residues[i]["CA"].get_vector()
                c = residues[i]["C"].get_vector()
                next_n = residues[i+1]["N"].get_vector()
                phi = float(np.degrees(calc_dihedral(prev_c, n, ca, c)))
                psi = float(np.degrees(calc_dihedral(n, ca, c, next_n)))
                allowed = (-180 < phi < 0 and -90 < psi < 180) or (-180 < phi < -30 and -60 < psi < 90)
                results.append({"residue": f"{residues[i].resname}{residues[i].id[1]}",
                                "chain": chain.id, "phi": round(phi, 2),
                                "psi": round(psi, 2), "outlier": not allowed})
            except (KeyError, TypeError):
                continue
    return results

def detect_clashes(structure, threshold=2.0):
    clashes = []
    atoms = list(structure.get_atoms())
    if len(atoms) > 3000: return []
    coords = np.array([a.coord for a in atoms])
    for i in range(len(atoms)):
        distances = np.linalg.norm(coords[i+1:] - coords[i], axis=1)
        for j, d in enumerate(distances):
            if d < threshold:
                clashes.append({
                    "atom1": f"{atoms[i].get_parent().resname}{atoms[i].get_parent().id[1]}:{atoms[i].name}",
                    "atom2": f"{atoms[i+1+j].get_parent().resname}{atoms[i+1+j].get_parent().id[1]}:{atoms[i+1+j].name}",
                    "distance": round(float(d), 2)})
                if len(clashes) >= 50: return clashes
    return clashes

def validate_structure(structure):
    phi_psi = calculate_phi_psi(structure)
    outliers = [r for r in phi_psi if r["outlier"]]
    clashes = detect_clashes(structure)
    score = max(0, 100 - len(outliers) * 2 - len(clashes) * 5)
    verdict = "✅ Structure valide" if score > 80 else ("⚠️ Corrections recommandées" if score > 50 else "❌ Structure à problème")
    return {"score_global": score, "total_residues": len(phi_psi),
            "outliers_ramachandran": len(outliers), "clashes_steriques": len(clashes),
            "details_outliers": outliers[:20], "details_clashes": clashes[:20],
            "verdict": verdict}


# ============================================================
# SECTION 5 : MORPHOMÉTRIE
# ============================================================

def extract_shape_descriptors(coords):
    if len(coords) < 3: return {}
    centroid = coords.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1))))
    G = ((coords - centroid).T @ (coords - centroid)) / len(coords)
    eigenvalues = np.sort(np.linalg.eigvalsh(G))[::-1]
    l1, l2, l3 = eigenvalues
    return {"rayon_giration": round(rg, 3),
            "asphéricité": round(float(l1 - 0.5 * (l2 + l3)), 4),
            "acylindricité": round(float(l2 - l3), 4),
            "anisotropie_forme": round(float((1.5 * (l1**2 + l2**2 + l3**2) / (l1 + l2 + l3)**2) - 0.5), 4) if (l1+l2+l3) > 0 else 0,
            "ellipticité": round(float(1 - (l3 / l1)), 4) if l1 > 0 else 0,
            "compacité": round(float(np.prod(np.sqrt(np.maximum(eigenvalues, 1e-6))) / ((4/3) * np.pi * rg**3)), 4) if rg > 0 else 0,
            "λ1": round(float(l1), 3), "λ2": round(float(l2), 3), "λ3": round(float(l3), 3),
            "ratio_axes": round(float(l1 / l3), 3) if l3 > 1e-6 else 0}

def fractal_dimension(coords, n_scales=10):
    if len(coords) < 10: return 0.0
    coords = coords - coords.min(axis=0)
    max_extent = coords.max()
    if max_extent < 1e-6: return 0.0
    scales = np.logspace(np.log10(max_extent / 100), np.log10(max_extent), n_scales)
    counts = [len(set(tuple((c // s).astype(int)) for c in coords)) for s in scales]
    log_s = np.log(1 / scales)
    log_c = np.log(np.array(counts) + 1e-9)
    return round(float(np.polyfit(log_s, log_c, 1)[0]), 3)

def extract_composition(structure):
    residues = [r for r in structure.get_residues() if r.id[0] == " "]
    if not residues: return {}
    resnames = [r.resname for r in residues]
    counts = Counter(resnames)
    total = len(resnames)
    freqs = {f"freq_{aa}": round(counts.get(aa, 0) / total, 4) for aa in AA_PROPERTIES}
    hydro, vol, charge, polar, arom, flex = [], [], [], [], [], []
    for aa in resnames:
        if aa in AA_PROPERTIES:
            h, v, c, p, a, f = AA_PROPERTIES[aa]
            hydro.append(h); vol.append(v); charge.append(c)
            polar.append(p); arom.append(a); flex.append(f)
    if not hydro: return {}
    freq_values = np.array(list(counts.values())) / total
    hydrophobic_aa = {'ALA','VAL','LEU','ILE','MET','PHE','TRP','PRO','CYS'}
    hydrophilic_aa = {'ARG','ASN','ASP','GLN','GLU','HIS','LYS','SER','THR','TYR'}
    n_hydro = sum(counts.get(aa, 0) for aa in hydrophobic_aa)
    n_philic = sum(counts.get(aa, 0) for aa in hydrophilic_aa)
    return {"nb_résidus": total,
            "entropie_composition": round(float(entropy(freq_values, base=2)), 3),
            "gravité_hydrophobicité": round(float(np.mean(hydro)), 3),
            "charge_nette": int(np.sum(charge)),
            "volume_moyen": round(float(np.mean(vol)), 2),
            "polarité_moyenne": round(float(np.mean(polar)), 3),
            "aromaticité": round(float(np.mean(arom)), 3),
            "flexibilité_moyenne": round(float(np.mean(flex)), 3),
            "ratio_hydrophobe_hydrophile": round(n_hydro / max(n_philic, 1), 3), **freqs}

def estimate_secondary_structure(structure):
    helix, sheet, coil = 0, 0, 0
    for chain in structure.get_chains():
        residues = [r for r in chain.get_residues() if r.id[0] == " "]
        for i in range(1, len(residues) - 1):
            try:
                prev_c = residues[i-1]["C"].get_vector()
                n = residues[i]["N"].get_vector()
                ca = residues[i]["CA"].get_vector()
                c = residues[i]["C"].get_vector()
                next_n = residues[i+1]["N"].get_vector()
                phi = np.degrees(calc_dihedral(prev_c, n, ca, c))
                psi = np.degrees(calc_dihedral(n, ca, c, next_n))
                if -100 < phi < -30 and -70 < psi < -10: helix += 1
                elif -160 < phi < -60 and 90 < psi < 180: sheet += 1
                else: coil += 1
            except (KeyError, TypeError):
                continue
    total = max(helix + sheet + coil, 1)
    return {"pct_hélice": round(100 * helix / total, 2),
            "pct_feuillet": round(100 * sheet / total, 2),
            "pct_coude": round(100 * coil / total, 2)}

def residue_network_features(structure, cutoff=8.0):
    ca_coords = [r["CA"].coord for chain in structure.get_chains()
                 for r in chain.get_residues() if r.id[0] == " " and "CA" in r]
    ca_coords = np.array(ca_coords)
    n = len(ca_coords)
    if n < 5: return {"degré_moyen": 0, "clustering_réseau": 0, "densité_réseau": 0, "nb_contacts": 0}
    dists = squareform(pdist(ca_coords))
    adj = (dists < cutoff) & (dists > 0)
    degrees = adj.sum(axis=1)
    triangles = np.trace(adj @ adj @ adj) / 6
    possible = n * (n - 1) * (n - 2) / 6
    return {"degré_moyen": round(float(degrees.mean()), 2),
            "clustering_réseau": round(float(3 * triangles / max(possible, 1)), 4),
            "densité_réseau": round(float(degrees.mean() / max(n - 1, 1)), 4),
            "nb_contacts": int(adj.sum() / 2)}

def compute_signature(structure, label="Protéine"):
    coords = np.array([a.coord for a in structure.get_atoms()])
    if len(coords) < 5: return None
    sig = {"label": label}
    sig.update(extract_shape_descriptors(coords))
    sig["dimension_fractale"] = fractal_dimension(coords)
    sig.update(extract_composition(structure))
    sig.update(estimate_secondary_structure(structure))
    sig.update(residue_network_features(structure))
    return sig

def detect_landmarks(structure, n_landmarks=20, radius=10.0):
    ca_list = [{"coord": r["CA"].coord.copy(), "resname": r.resname,
                "resid": f"{chain.id}:{r.id[1]}"}
               for chain in structure.get_chains()
               for r in chain.get_residues() if r.id[0] == " " and "CA" in r]
    if len(ca_list) < n_landmarks: return []
    coords = np.array([x["coord"] for x in ca_list])
    scores = []
    for i, c in enumerate(coords):
        dists = np.linalg.norm(coords - c, axis=1)
        neighbors = np.where((dists > 0) & (dists < radius))[0]
        if len(neighbors) < 3: scores.append(0); continue
        local_std = np.std(dists[neighbors])
        hyd = AA_PROPERTIES.get(ca_list[i]["resname"], (0,)*6)[0]
        scores.append(local_std * (1 + abs(hyd) / 5))
    top_idx = np.argsort(scores)[::-1][:n_landmarks]
    return [{"id": ca_list[i]["resid"], "resname": ca_list[i]["resname"],
             "x": round(float(coords[i][0]), 2), "y": round(float(coords[i][1]), 2),
             "z": round(float(coords[i][2]), 2), "score": round(float(scores[i]), 3)}
            for i in top_idx]

def compute_morpho_topology_index(sig):
    if not sig: return {}
    compactness = sig.get("compacité", 0); ellipticity = sig.get("ellipticité", 0)
    clustering = sig.get("clustering_réseau", 0)
    degree = sig.get("degré_moyen", 0); density = sig.get("densité_réseau", 0)
    form_axis = min(1.0, (ellipticity + (1 - compactness)) / 2)
    topo_axis = min(1.0, (degree / 15 + clustering * 2 + density) / 3)
    morpho_topo = round(0.5 * form_axis + 0.5 * topo_axis, 4)
    if form_axis > 0.6 and topo_axis > 0.6: classe = "Allongée-Hub"
    elif form_axis > 0.6: classe = "Allongée-Distribuée"
    elif topo_axis > 0.6: classe = "Globulaire-Hub"
    else: classe = "Globulaire-Distribuée"
    return {"forme_axe": round(form_axis, 4), "topologie_axe": round(topo_axis, 4),
            "indice_morpho_topologique": morpho_topo, "classification_morpho_topo": classe}

def compute_druggability_index(sig):
    if not sig: return {}
    compactness = sig.get("compacité", 0); fractal = sig.get("dimension_fractale", 0)
    gravy = sig.get("gravité_hydrophobicité", 0); arom = sig.get("aromaticité", 0)
    polar = sig.get("polarité_moyenne", 0); clustering = sig.get("clustering_réseau", 0)
    ratio_hp = sig.get("ratio_hydrophobe_hydrophile", 0)
    s_compact = max(0, min(1, 1 - abs(compactness - 0.5) * 2))
    s_fractal = max(0, min(1, 1 - abs(fractal - 2.3) / 0.5))
    s_gravy = max(0, min(1, 1 - abs(gravy) / 2))
    s_arom = max(0, min(1, 1 - abs(arom - 0.1) * 5))
    s_polar = max(0, min(1, 1 - abs(polar - 0.3) * 2))
    s_cluster = max(0, min(1, 1 - abs(clustering - 0.15) * 4))
    s_hp = max(0, min(1, 1 - abs(ratio_hp - 1.5) / 3))
    weights = [0.15, 0.15, 0.20, 0.10, 0.10, 0.15, 0.15]
    scores = [s_compact, s_fractal, s_gravy, s_arom, s_polar, s_cluster, s_hp]
    index = round(sum(w * s for w, s in zip(weights, scores)), 4)
    if index > 0.75: classe = "🟢 Fortement druggable"
    elif index > 0.5: classe = "🟡 Modérément druggable"
    elif index > 0.25: classe = "🟠 Faiblement druggable"
    else: classe = "🔴 Peu probable"
    return {"indice_druggabilité": index, "classification_druggabilité": classe,
            "score_compacité": round(s_compact, 3), "score_fractal": round(s_fractal, 3),
            "score_hydrophobicité": round(s_gravy, 3), "score_aromaticité": round(s_arom, 3),
            "score_polarité": round(s_polar, 3), "score_clustering": round(s_cluster, 3),
            "score_ratio_HP": round(s_hp, 3)}


# ============================================================
# SECTION 6 : STATISTIQUES & ML
# ============================================================

def build_feature_matrix(signatures):
    df = pd.DataFrame(signatures).set_index("label")
    return df.select_dtypes(include=[np.number]).fillna(0)

def run_pca(fm, n=2):
    if len(fm) < 2: return None, None
    X = StandardScaler().fit_transform(fm)
    pca = PCA(n_components=min(n, X.shape[0], X.shape[1]))
    return pca.fit_transform(X), pca

def run_kmeans(fm, k=3):
    if len(fm) < k: k = max(2, len(fm))
    X = StandardScaler().fit_transform(fm)
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return 0.0
    dof = nx + ny - 2
    pooled = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / dof)
    if pooled < 1e-9: return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)

def statistical_test_features(df, groups, feature_cols):
    ga, gb = groups[0], groups[1]
    mask_a = df["_group"] == ga; mask_b = df["_group"] == gb
    results = []
    for feat in feature_cols:
        x = df.loc[mask_a, feat].dropna().values
        y = df.loc[mask_b, feat].dropna().values
        if len(x) < 2 or len(y) < 2: continue
        try: _, p = ttest_ind(x, y, equal_var=False)
        except Exception: p = 1.0
        d = cohen_d(x, y)
        direction = ga if np.mean(x) > np.mean(y) else gb
        abs_d = abs(d)
        mag = "🟢 Large" if abs_d >= 0.8 else "🟡 Moyenne" if abs_d >= 0.5 else "🟠 Petite" if abs_d >= 0.2 else "⚪ Négligeable"
        results.append({"Feature": feat, "Moy "+ga: round(float(np.mean(x)), 4),
                       "Moy "+gb: round(float(np.mean(y)), 4), "p-value": p,
                       "Cohen's d": round(d, 3), "Direction": direction,
                       "Taille d'effet": mag, "-log10(p)": round(-np.log10(max(p, 1e-300)), 2)})
    df_res = pd.DataFrame(results).sort_values("p-value")
    if len(df_res) > 0:
        _, fdr, _, _ = multipletests(df_res["p-value"].values, method="fdr_bh")
        df_res["FDR"] = fdr
    return df_res

def rank_biomarkers(df_stats):
    if df_stats is None or len(df_stats) == 0: return None
    r = df_stats.copy()
    r["abs_d"] = r["Cohen's d"].abs()
    def norm(s):
        if s.max() - s.min() < 1e-9: return pd.Series([0]*len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min())
    r["Score biomarqueur"] = (0.5 * norm(r["abs_d"]) + 0.5 * norm(r["-log10(p)"])).round(4)
    return r.sort_values("Score biomarqueur", ascending=False)


# ============================================================
# SECTION 7 : DOCKING
# ============================================================

def run_docking(protein_pdb, smiles, center, box):
    if not PANDADOCK_AVAILABLE:
        np.random.seed(hash(smiles) % 2**32)
        return [{"Score (kcal/mol)": round(-8.5 + np.random.rand() * 3, 2),
                 "Pose": i + 1, "Ligand PDB": "REMARK Simulation\nEND"} for i in range(10)]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tp:
        tp.write(protein_pdb.encode()); p = tp.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as tl:
        tl.write(smiles.encode()); l = tl.name
    try:
        pp = Protein(p); pl = Ligand(l)
        res = pp.dock(ligand=pl, center=center, box_size=box,
                      scoring_function=CompositeScoringFunction(), num_poses=10)
        return [{"Score (kcal/mol)": x.score, "Pose": x.pose_id, "Ligand PDB": x.to_pdb()} for x in res.poses]
    finally:
        os.unlink(p); os.unlink(l)

def detect_cavities(structure, min_volume=100, max_cavities=5):
    atoms = list(structure.get_atoms())
    if len(atoms) < 50: return []
    coords = np.array([a.coord for a in atoms])
    centroid = coords.mean(axis=0)
    max_extent = np.max(np.linalg.norm(coords - centroid, axis=1))
    grid_size = 30
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=int)
    box_size = 2 * max_extent * 1.1
    cell_size = box_size / grid_size
    origin = centroid - box_size / 2
    for c in coords:
        idx = ((c - origin) / cell_size).astype(int)
        if all(0 <= i < grid_size for i in idx):
            grid[tuple(idx)] = 1
    cavities = []
    visited = np.zeros_like(grid, dtype=bool)
    for x in range(2, grid_size - 2):
        for y in range(2, grid_size - 2):
            for z in range(2, grid_size - 2):
                if grid[x, y, z] == 0 and not visited[x, y, z]:
                    region = []
                    stack = [(x, y, z)]
                    touches_border = False
                    while stack and len(region) < 200:
                        cx, cy, cz = stack.pop()
                        if not (0 <= cx < grid_size and 0 <= cy < grid_size and 0 <= cz < grid_size): continue
                        if visited[cx, cy, cz] or grid[cx, cy, cz] == 1: continue
                        if cx < 2 or cx > grid_size-3 or cy < 2 or cy > grid_size-3 or cz < 2 or cz > grid_size-3:
                            touches_border = True
                        visited[cx, cy, cz] = True
                        region.append((cx, cy, cz))
                        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                            stack.append((cx+dx, cy+dy, cz+dz))
                    if not touches_border and len(region) >= 10:
                        region_coords = np.array(region)
                        center_xyz = origin + region_coords.mean(axis=0) * cell_size
                        volume = len(region) * (cell_size ** 3)
                        if volume >= min_volume:
                            cavities.append({"center": tuple(center_xyz), "volume_A3": round(volume, 1)})
    cavities = sorted(cavities, key=lambda c: -c["volume_A3"])[:max_cavities]
    for cav in cavities:
        v = cav["volume_A3"]
        cav["druggabilité"] = "🟢 Idéale" if 300 <= v <= 1000 else "🟡 Acceptable" if (100 <= v < 300 or 1000 < v <= 2000) else "🔴 Peu favorable"
    return cavities

def identify_interface_residues(sa, sb, cutoff=6.0):
    from scipy.spatial import cKDTree
    atoms_a = [(a.coord, a.get_parent().resname, a.get_parent().id[1], a.get_parent().get_parent().id) for a in sa.get_atoms()]
    atoms_b = [(a.coord, a.get_parent().resname, a.get_parent().id[1], a.get_parent().get_parent().id) for a in sb.get_atoms()]
    coords_b = np.array([x[0] for x in atoms_b])
    if len(atoms_a) > 2000 or len(atoms_b) > 2000: return [], []
    tree_b = cKDTree(coords_b)
    ia, ib = set(), set()
    for i, (ca, res_a, id_a, ch_a) in enumerate(atoms_a):
        for j in tree_b.query_ball_point(ca, cutoff):
            ia.add(f"{ch_a}:{res_a}{id_a}")
            ib.add(f"{atoms_b[j][3]}:{atoms_b[j][1]}{atoms_b[j][2]}")
    return sorted(ia), sorted(ib)


# ============================================================
# SECTION 8 : TRANSCRIPTOMIQUE
# ============================================================

def simulate_expression_matrix(n_genes=300, n_samples=20, seed=42):
    np.random.seed(seed)
    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    n_ctrl = n_samples // 2
    n_case = n_samples - n_ctrl
    sample_names = [f"Ctrl_{i+1}" for i in range(n_ctrl)] + [f"Case_{i+1}" for i in range(n_case)]
    base = np.random.lognormal(mean=5, sigma=1.5, size=(n_genes, n_samples))
    for idx in np.random.choice(n_genes, int(n_genes * 0.1), replace=False):
        fc = np.random.uniform(1.5, 4.0) * np.random.choice([-1, 1])
        base[idx, n_ctrl:] *= np.exp(fc / 2)
    return pd.DataFrame(base, index=gene_names, columns=sample_names), \
           pd.Series(["Contrôle"] * n_ctrl + ["Malade"] * n_case, index=sample_names, name="Groupe")

def differential_expression_analysis(df_expr, groups, log2fc=1.0, fdr_th=0.05):
    ctrl = groups[groups == groups.unique()[0]].index.tolist()
    case = groups[groups == groups.unique()[1]].index.tolist()
    results = []
    for gene in df_expr.index:
        c_vals = df_expr.loc[gene, ctrl].values
        m_vals = df_expr.loc[gene, case].values
        if len(c_vals) < 2 or len(m_vals) < 2: continue
        try: _, p = ttest_ind(m_vals, c_vals, equal_var=False)
        except Exception: continue
        mean_c = np.mean(c_vals) + 1e-9; mean_m = np.mean(m_vals) + 1e-9
        results.append({"Gène": gene, "log2FC": round(float(np.log2(mean_m / mean_c)), 4),
                       "p-value": float(p), "Moyenne contrôle": round(float(mean_c), 3),
                       "Moyenne malade": round(float(mean_m), 3)})
    df_res = pd.DataFrame(results)
    if len(df_res) == 0: return None
    _, fdr, _, _ = multipletests(df_res["p-value"].values, method="fdr_bh")
    df_res["FDR"] = fdr
    df_res["DE significatif"] = (df_res["FDR"] < fdr_th) & (df_res["log2FC"].abs() >= log2fc)
    return df_res.sort_values("p-value")

def simulate_gepia2_expression(gene, n_tumor=100, n_normal=50, seed=42):
    np.random.seed(seed + hash(gene) % 10000)
    base_normal = np.random.lognormal(mean=4, sigma=1.0, size=n_normal)
    fc = np.random.uniform(0.3, 3.5)
    base_tumor = np.random.lognormal(mean=4 + np.log(fc), sigma=1.2, size=n_tumor)
    return pd.DataFrame({"Expression": np.concatenate([base_normal, base_tumor]),
                        "Type": ["Normal"] * n_normal + ["Tumeur"] * n_tumor, "Gène": gene})

def simulate_ualcan_subgroups(gene, cancer="BRCA", subgroup="stage", seed=42):
    np.random.seed(seed + hash(gene + cancer + subgroup) % 10000)
    subgroups = {"stage": ["Stage I", "Stage II", "Stage III", "Stage IV"],
                 "grade": ["Grade 1", "Grade 2", "Grade 3", "Grade 4"],
                 "age": ["21-40", "41-60", "61-80", "81-100"],
                 "gender": ["Male", "Female"],
                 "race": ["Caucasian", "African-American", "Asian"]}
    cats = subgroups.get(subgroup, ["A", "B", "C"])
    rows = []
    for cat in cats:
        n = np.random.randint(20, 80)
        fc = 1 + cats.index(cat) * np.random.uniform(0.3, 0.8)
        for e in np.random.lognormal(mean=4 + np.log(fc), sigma=0.9, size=n):
            rows.append({"Subgroupe": cat, "Expression": e, "Gène": gene, "Cancer": cancer})
    return pd.DataFrame(rows)

def prepare_ml_data(df_de):
    if df_de is None or len(df_de) < 10: return None
    df = df_de.copy().dropna(subset=["log2FC", "FDR"])
    df["feature_1"] = df["log2FC"].abs()
    df["feature_2"] = -np.log10(df["FDR"].clip(lower=1e-300))
    df["feature_3"] = df["Moyenne malade"] - df["Moyenne contrôle"]
    df["feature_4"] = df["log2FC"] * df["feature_2"]
    return df

def train_xgboost_classifier(df_ml, top_n=50):
    if not XGBOOST_AVAILABLE or df_ml is None or len(df_ml) < top_n: return None
    df_top = df_ml.nlargest(top_n, "feature_4")
    X = df_top[["feature_1", "feature_2", "feature_3", "feature_4"]].values
    y = (df_top["log2FC"] > 0).astype(int).values
    if len(np.unique(y)) < 2: return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                              random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test); y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {"Accuracy": round(accuracy_score(y_test, y_pred), 4),
               "AUC": round(roc_auc_score(y_test, y_proba), 4),
               "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
               "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
               "F1": round(f1_score(y_test, y_pred, zero_division=0), 4)}
    importance = pd.DataFrame({"Feature": ["|log2FC|", "-log10(FDR)", "Δ Expression", "Score composite"],
                              "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    return {"metrics": metrics, "importance": importance}


# ============================================================
# SECTION 9 : INGÉNIERIE DES PROTÉINES
# ============================================================

def classify_mutation_type(old_aa, new_aa):
    if old_aa not in AA_PROPERTIES or new_aa not in AA_PROPERTIES: return "Inconnu"
    old_h, _, old_c, old_p, old_a, _ = AA_PROPERTIES[old_aa]
    new_h, _, new_c, new_p, new_a, _ = AA_PROPERTIES[new_aa]
    if old_c != new_c: return "Modification de charge"
    if abs(old_h - new_h) > 2: return "Modification d'hydrophobicité"
    if old_a != new_a: return "Modification d'aromaticité"
    if abs(old_p - new_p) > 0.5: return "Modification de polarité"
    return "Substitution conservative"

def predict_mutation_impact(structure, mutation):
    try:
        chain = structure[0][mutation["Chaîne"]]
        residue = chain[(" ", mutation["Position"], " ")]
    except (KeyError, Exception):
        return None
    if "CA" not in residue: return None
    ca_coord = residue["CA"].coord
    accessibility = sum(1 for oc in structure.get_chains() for ores in oc.get_residues()
                        if ores.id[0] == " " and ores != residue and "CA" in ores
                        and np.linalg.norm(ores["CA"].coord - ca_coord) < 8.0)
    old_aa, new_aa = mutation["Original"], mutation["Mutant"]
    if old_aa in AA_PROPERTIES and new_aa in AA_PROPERTIES:
        dh = AA_PROPERTIES[new_aa][0] - AA_PROPERTIES[old_aa][0]
        dv = AA_PROPERTIES[new_aa][1] - AA_PROPERTIES[old_aa][1]
        bf = max(0, min(1, 1 - (accessibility / 30)))
        ddg = dv * 0.01 * bf + abs(dh) * 0.3
        if abs(dh) > 2 and accessibility > 15: ddg += 0.5
    else: ddg = 0.0
    return {"mutation": f"{mutation['Original']}{mutation['Position']}{mutation['Mutant']}",
            "accessibilité": accessibility, "ΔΔG_prédit": round(ddg, 2),
            "stabilité": "🟢 Stabilisante" if ddg < -0.5 else "🔴 Déstabilisante" if ddg > 0.5 else "🟡 Neutre"}

def saturation_mutagenesis(structure, chain_id, position):
    try:
        residue = structure[0][chain_id][(" ", position, " ")]
        original_aa = residue.resname
    except Exception: return None
    results = []
    for new_aa in AA_PROPERTIES.keys():
        if new_aa == original_aa: continue
        impact = predict_mutation_impact(structure, {"Chaîne": chain_id, "Position": position,
                                                      "Original": original_aa, "Mutant": new_aa})
        if impact:
            results.append({"Mutation": f"{original_aa}{position}{new_aa}",
                           "ΔΔG (kcal/mol)": impact["ΔΔG_prédit"], "Stabilité": impact["stabilité"]})
    df = pd.DataFrame(results)
    return df.sort_values("ΔΔG (kcal/mol)") if len(df) > 0 else df

def propose_stabilizing_mutations(structure, top_n=10):
    proposals = []
    for chain in structure.get_chains():
        for residue in chain.get_residues():
            if residue.id[0] != " " or "CA" not in residue: continue
            ca = residue["CA"].coord
            n = sum(1 for a in structure.get_atoms() if a.get_parent() != residue
                    and np.linalg.norm(a.coord - ca) < 6.0)
            r, p = residue.resname, residue.id[1]
            if r == "GLY" and n > 20:
                proposals.append({"Mutation": f"G{p}A", "Chaîne": chain.id, "Raison": "Gly enfoui → Ala", "Score": 0.7})
            elif r == "LYS" and n < 15:
                proposals.append({"Mutation": f"K{p}R", "Chaîne": chain.id, "Raison": "Lys surface → Arg", "Score": 0.5})
            elif r == "SER" and n > 18:
                proposals.append({"Mutation": f"S{p}A", "Chaîne": chain.id, "Raison": "Ser → Ala", "Score": 0.6})
    df = pd.DataFrame(proposals)
    return df.sort_values("Score", ascending=False).head(top_n) if len(df) > 0 else df

def identify_disulfide_opportunities(structure):
    cys = [{"chain": chain.id, "pos": r.id[1], "coord": r["SG"].coord}
           for chain in structure.get_chains()
           for r in chain.get_residues() if r.id[0] == " " and r.resname == "CYS" and "SG" in r]
    if len(cys) < 2: return pd.DataFrame()
    pairs = []
    for i in range(len(cys)):
        for j in range(i + 1, len(cys)):
            d = np.linalg.norm(cys[i]["coord"] - cys[j]["coord"])
            if 4.0 <= d <= 8.0:
                pairs.append({"Cys 1": f"{cys[i]['chain']}:CYS{cys[i]['pos']}",
                             "Cys 2": f"{cys[j]['chain']}:CYS{cys[j]['pos']}",
                             "Distance S-S (Å)": round(float(d), 2),
                             "Faisabilité": "🟢 Excellente" if 5.0 <= d <= 7.0 else "🟡 Acceptable"})
    return pd.DataFrame(pairs)

def generate_variant_library(structure, positions):
    library = []
    for chain_id, pos in positions:
        try:
            original = structure[0][chain_id][(" ", pos, " ")].resname
        except Exception: continue
        for new_aa in AA_PROPERTIES.keys():
            if new_aa == original: continue
            impact = predict_mutation_impact(structure, {"Chaîne": chain_id, "Position": pos,
                                                          "Original": original, "Mutant": new_aa})
            if impact:
                library.append({"Variant_ID": f"{original}{pos}{new_aa}", "Chaîne": chain_id,
                               "Position": pos, "Original": original, "Mutant": new_aa,
                               "Type": classify_mutation_type(original, new_aa),
                               "ΔΔG (kcal/mol)": impact["ΔΔG_prédit"], "Stabilité": impact["stabilité"]})
    return pd.DataFrame(library)

def aa3to1(aa3):
    m = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
         "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
         "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
    return m.get(aa3, "X")

def reverse_complement(seq):
    c = {"A":"T","T":"A","G":"C","C":"G"}
    return "".join(c.get(b, "N") for b in reversed(seq))

def generate_lab_protocol(structure, mutations, enzyme="Q5"):
    seq = "".join(aa3to1(r.resname) for chain in structure.get_chains()
                  for r in chain.get_residues() if r.id[0] == " " and r.resname in AA_PROPERTIES)
    primers = []
    for mut in mutations:
        pos = mut["Position"]
        if pos < 1 or pos > len(seq): continue
        old = aa3to1(mut["Original"]); new = aa3to1(mut["Mutant"])
        start = max(0, pos - 1 - 18); end = min(len(seq), pos - 1 + 19)
        fwd = seq[start:pos-1] + new + seq[pos:end]
        rev = reverse_complement(fwd)
        primers.append({"Mutation": f"{old}{pos}{new}", "Fwd": fwd, "Rev": rev,
                       "Tm": round(64.9 + 41 * (fwd.count("G") + fwd.count("C") - 16.4) / len(fwd), 1)})
    protocol = f"# PROTOCOLE DE MUTAGENÈSE DIRIGÉE\n\n## Protéine : {structure.header.get('name', 'N/A')}\n\n## Mutations\n"
    for m in mutations: protocol += f"- {m['Original']}{m['Position']}{m['Mutant']}\n"
    protocol += "\n## Amorces\n| Mutation | Fwd | Rev | Tm (°C) |\n|----------|-----|-----|---------|\n"
    for p in primers: protocol += f"| {p['Mutation']} | {p['Fwd']} | {p['Rev']} | {p['Tm']} |\n"
    protocol += f"\n## PCR ({enzyme})\n- 95°C 30s, 18× (95°C 30s, 55°C 1min, 68°C 1min/kb), 68°C 10min\n- DpnI 1h à 37°C, transformer\n"
    return protocol, primers


# ============================================================
# SECTION 10 : COPILOTE IA (outils)
# ============================================================

def build_copilot_tools():
    return [
        {"type": "function", "function": {"name": "fetch_pdb", "description": "Télécharge une PDB",
            "parameters": {"type": "object", "properties": {"pdb_id": {"type": "string"}}, "required": ["pdb_id"]}}},
        {"type": "function", "function": {"name": "get_structure_info", "description": "Infos structure"}},
        {"type": "function", "function": {"name": "run_validation", "description": "Validation"}},
        {"type": "function", "function": {"name": "compute_morphometrics", "description": "Signature"}},
        {"type": "function", "function": {"name": "compute_druggability", "description": "Druggabilité"}},
    ]

def execute_copilot_tool(name, args):
    try:
        if name == "fetch_pdb":
            txt = fetch_pdb_structure(args["pdb_id"])
            st.session_state.pdb_text = txt
            st.session_state.structure = parse_pdb_structure(txt)
            return f"Structure {args['pdb_id']} chargée."
        if name == "get_structure_info":
            return json.dumps(get_structure_info(st.session_state.structure), indent=2) if st.session_state.structure else "Aucune."
        if name == "run_validation":
            return json.dumps(validate_structure(st.session_state.structure), indent=2, default=str) if st.session_state.structure else "Aucune."
        if name == "compute_morphometrics":
            if st.session_state.structure:
                sig = compute_signature(st.session_state.structure, "Copilot")
                st.session_state.last_signature = sig
                return f"Rg={sig.get('rayon_giration')} Å"
            return "Échec."
        if name == "compute_druggability":
            return json.dumps(compute_druggability_index(st.session_state.last_signature), indent=2) if st.session_state.last_signature else "Signature requise."
    except Exception as e:
        return f"Erreur : {e}"
    return "Inconnu."


# ============================================================
# SECTION 10B : MODULE MULTI-LLM (v14)
# ============================================================

LLM_PROVIDERS = {
    "OpenAI (payant)": {
        "category": "☁️ API Cloud",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini",
        "secret_key": "openai", "requires_key": True, "free": False,
        "rate_limit": "Payant",
        "description": "Haute performance, multimodal"
    },
    "Groq (gratuit)": {
        "category": "☁️ API Cloud",
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                   "mixtral-8x7b-32768", "gemma2-9b-it"],
        "default_model": "llama-3.3-70b-versatile",
        "secret_key": "groq", "requires_key": True, "free": True,
        "rate_limit": "30 req/min, 14 400 req/jour",
        "description": "Le plus rapide, gratuit, compatible OpenAI"
    },
    "Google Gemini (gratuit)": {
        "category": "☁️ API Cloud",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "default_model": "gemini-2.5-flash",
        "secret_key": "gemini", "requires_key": True, "free": True,
        "rate_limit": "5-20 req/min, 20-50 req/jour",
        "description": "Multimodal, gratuit"
    },
    "Mistral (gratuit)": {
        "category": "☁️ API Cloud",
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-small-latest", "mistral-large-latest", "codestral-latest"],
        "default_model": "mistral-small-latest",
        "secret_key": "mistral", "requires_key": True, "free": True,
        "rate_limit": "1 req/sec, 1B tokens/mois",
        "description": "Gratuit, bon pour le code"
    },
    "Cerebras (gratuit)": {
        "category": "☁️ API Cloud",
        "base_url": "https://api.cerebras.ai/v1",
        "models": ["llama3.3-70b", "llama3.1-8b", "qwen-3-32b"],
        "default_model": "llama3.3-70b",
        "secret_key": "cerebras", "requires_key": True, "free": True,
        "rate_limit": "30 req/min, 1M tokens/jour",
        "description": "Ultra-rapide, 1M tokens/jour gratuits"
    },
    "OpenRouter (gratuit)": {
        "category": "☁️ API Cloud",
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["meta-llama/llama-3.3-70b-instruct:free",
                   "mistralai/mistral-small-3.2-24b-instruct:free",
                   "google/gemma-2-9b-it:free",
                   "qwen/qwen-2.5-72b-instruct:free"],
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "secret_key": "openrouter", "requires_key": True, "free": True,
        "rate_limit": "20 req/min, 50 req/jour",
        "description": "Accès multi-modèles via une seule API"
    },
    "Ollama (local)": {
        "category": "💻 Local",
        "base_url": "http://localhost:11434/v1",
        "models": ["llama3.2", "llama3.1", "mistral", "qwen2.5",
                   "phi3", "gemma2", "codellama"],
        "default_model": "llama3.2",
        "secret_key": None, "requires_key": False, "free": True,
        "rate_limit": "Illimité (local)",
        "description": "100% local, confidentiel, aucune limite"
    },
    "Hugging Face (spécialisé)": {
        "category": "🧬 Spécialisé",
        "base_url": "https://api-inference.huggingface.co/v1",
        "models": ["ContactDoctor/Bio-Medical-Llama-3-8B",
                   "ncbi/Gene-R1-8B",
                   "microsoft/BioGPT-Large"],
        "default_model": "ContactDoctor/Bio-Medical-Llama-3-8B",
        "secret_key": "huggingface", "requires_key": True, "free": True,
        "rate_limit": "Limité (Hugging Face Inference API)",
        "description": "Modèles fine-tunés pour la biologie"
    }
}


def get_llm_client(provider_name):
    if provider_name not in LLM_PROVIDERS:
        return None, "Fournisseur inconnu."
    config = LLM_PROVIDERS[provider_name]

    if not config["requires_key"]:
        try:
            base_url = config["base_url"]
            try:
                base_url = st.secrets["ollama"]["base_url"]
            except (KeyError, Exception):
                pass
            return OpenAI(api_key="ollama", base_url=base_url), None
        except Exception as e:
            return None, f"Erreur Ollama : {e}"

    try:
        api_key = st.secrets[config["secret_key"]]["api_key"]
        if not api_key or api_key.startswith("VOTRE_"):
            return None, f"Clé API manquante pour {provider_name}."
        return OpenAI(api_key=api_key, base_url=config["base_url"]), None
    except KeyError:
        return None, f"Configuration manquante : `[{config['secret_key']}]` dans secrets.toml"
    except Exception as e:
        return None, f"Erreur : {e}"


def run_multi_llm_chat(user_message, history, provider_name,
                       model_override=None, system_prompt=None,
                       tools=None, temperature=0.3, max_tokens=2000):
    client, error = get_llm_client(provider_name)
    if error:
        return f"⚠️ {error}"

    config = LLM_PROVIDERS[provider_name]
    model = model_override or config["default_model"]

    if system_prompt is None:
        system_prompt = """Tu es BioStruct Copilot, un assistant expert en bioinformatique 
structurale, morphométrie des protéines et ingénierie des protéines. 
Réponds en français, de manière concise et scientifique."""

    messages = [{"role": "system", "content": system_prompt}] + history + \
               [{"role": "user", "content": user_message}]

    try:
        kwargs = {"model": model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens}
        if tools and provider_name in ("OpenAI (payant)", "Groq (gratuit)"):
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = execute_copilot_tool(tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            final = client.chat.completions.create(model=model, messages=messages)
            return final.choices[0].message.content
        return msg.content
    except Exception as e:
        return f"❌ Erreur avec {provider_name} ({model}) : {e}"


def check_ollama_available():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            return True, [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return False, []


SPECIALIZED_MODELS = {
    "Bio-Medical-Llama-3-8B": {
        "hf_id": "ContactDoctor/Bio-Medical-Llama-3-8B",
        "description": "LLM médical fine-tuné",
        "specialty": "Médecine, pharmacologie",
        "size": "8B params", "provider": "Hugging Face"
    },
    "Gene-R1-8B": {
        "hf_id": "ncbi/Gene-R1-8B",
        "description": "LLM pour l'analyse de gènes",
        "specialty": "Génomique", "size": "8B params",
        "provider": "NCBI / Hugging Face"
    },
    "BioGPT-Large": {
        "hf_id": "microsoft/BioGPT-Large",
        "description": "GPT-2 fine-tuné biomédical",
        "specialty": "Extraction d'information", "size": "1.5B params",
        "provider": "Microsoft"
    }
}


def query_specialized_model(model_name, prompt, max_tokens=500):
    if model_name not in SPECIALIZED_MODELS: return "Modèle inconnu."
    info = SPECIALIZED_MODELS[model_name]
    try:
        api_key = st.secrets["huggingface"]["api_key"]
        if not api_key or api_key.startswith("VOTRE_"):
            return "⚠️ Clé Hugging Face manquante."
    except (KeyError, Exception):
        return "⚠️ Configuration Hugging Face manquante."

    url = f"https://api-inference.huggingface.co/models/{info['hf_id']}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": 0.3}}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            result = r.json()
            if isinstance(result, list) and result:
                return result[0].get("generated_text", str(result))
            return str(result)
        return f"❌ Erreur {r.status_code} : {r.text[:200]}"
    except Exception as e:
        return f"❌ Erreur : {e}"


# ============================================================
# SECTION 11 : INTERFACE STREAMLIT
# ============================================================

st.set_page_config(page_title="BioStruct AI v14", page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")

defaults = {
    "user": None, "pdb_text": None, "structure": None,
    "pdb_text_b": None, "structure_b": None,
    "docking_results": [], "validation_report": None,
    "copilot_history": [], "mode": "Débutant",
    "signatures_library": {}, "last_signature": None,
    "landmarks": [], "multi_scale_results": [],
    "expression_matrix": None, "expression_groups": None, "de_results": None,
    "survival_data": None, "survival_result": None, "ml_result": None,
    "gepia_data": None, "ualcan_data": None,
    "blind_docking_results": None, "cavities": None,
    "mutation_results": None, "saturation_results": None,
    "stabilization_results": None, "disulfide_results": None,
    "variant_library": None, "lab_protocol": None, "lab_primers": None,
    "md_frames": None, "ppi_results": None, "biomarker_results": None,
    "llm_provider": "Groq (gratuit)", "llm_model": None,
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

with st.sidebar:
    st.title("🧬 BioStruct AI v14")
    st.session_state.mode = st.radio("🎓 Mode", ["Débutant", "Expert"], horizontal=True)
    st.markdown("---")
    if st.session_state.user is None:
        email = st.text_input("Email", placeholder="votre.email@exemple.com")
        if st.button("Connexion / Inscription", use_container_width=True):
            if email and "@" in email:
                user = get_or_create_user(email)
                if user: st.session_state.user = user; st.rerun()
    else:
        st.success(f"👤 {st.session_state.user['email']}")
        if st.button("Déconnexion", use_container_width=True):
            st.session_state.user = None; st.rerun()
    st.markdown("---")
    page = st.radio("Module", [
        "🏠 Accueil",
        "🤖 Configuration LLM",
        "🤖 Copilote Multi-LLM",
        "🧬 Modèles spécialisés",
        "🔬 Visualisation 3D",
        "✅ Validation",
        "🧪 Docking",
        "🎯 Docking aveugle",
        "🔗 Docking PPI",
        "🧬 Morphométrie",
        "🎯 Landmarks auto",
        "🔗 Morpho-topologie",
        "💊 Druggabilité",
        "🌍 Variants génétiques",
        "🌡️ Impact environnemental",
        "🌐 Comparaison populations",
        "🔬 Découverte biomarqueurs",
        "📊 Transcriptomique",
        "📈 Analyse de survie",
        "🤖 Prédiction ML",
        "📊 GEPIA2 (TCGA)",
        "🧪 UALCAN (sous-groupes)",
        "🧪 Mutagenèse in silico",
        "🔧 Saturation mutagenèse",
        "🛡️ Stabilisation (PROSS-like)",
        "🔗 Ponts disulfures",
        "📚 Bibliothèque de variants",
        "🎨 Visualisation 3D mutations",
        "📋 Protocole laboratoire",
        "📊 Mes Analyses"
    ], label_visibility="collapsed")


# ============================================================
# PAGES
# ============================================================

if page == "🏠 Accueil":
    st.title("🧬 BioStruct AI v14.0")
    st.markdown(f"""
    ### Plateforme Intégrée de Bioinformatique Structurale
    **Mode : {st.session_state.mode}** | **Multi-LLM intégré**

    #### Options LLM disponibles
    - ☁️ **API Cloud gratuites** : Groq, Gemini, Mistral, Cerebras, OpenRouter
    - 💻 **Local** : Ollama (confidentiel, illimité)
    - 🧬 **Spécialisé** : Bio-Medical-Llama, Gene-R1-8B, BioGPT

    #### Modules disponibles
    - 🔬 Analyse structurale (visualisation, validation, docking)
    - 🧬 Morphométrie (signature, landmarks, druggabilité)
    - 🌐 Génomique (variants, populations)
    - 📊 Transcriptomique (DE, ML)
    - 🧪 Ingénierie des protéines (mutagenèse, stabilisation)
    """)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modules", "30+"); c2.metric("Version", "14.0.0")
    c3.metric("Signatures", len(st.session_state.signatures_library))
    c4.metric("LLM actif", st.session_state.llm_provider)


elif page == "🤖 Configuration LLM":
    st.title("🤖 Configuration Multi-LLM")
    st.markdown("### 📊 Statut des fournisseurs")

    status_rows = []
    for name, cfg in LLM_PROVIDERS.items():
        if not cfg["requires_key"]:
            available, _ = check_ollama_available()
            status = "✅ Disponible" if available else "❌ Non lancé"
        else:
            try:
                key = st.secrets[cfg["secret_key"]]["api_key"]
                status = "✅ Configuré" if key and not key.startswith("VOTRE_") else "⚠️ Clé manquante"
            except (KeyError, Exception):
                status = "⚠️ Non configuré"
        status_rows.append({"Fournisseur": name, "Catégorie": cfg["category"],
                           "Gratuit": "✅" if cfg["free"] else "💰",
                           "Statut": status, "Limite": cfg["rate_limit"]})
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🎯 Sélection du fournisseur actif")
    categories = ["☁️ API Cloud", "💻 Local", "🧬 Spécialisé"]
    selected_cat = st.selectbox("Catégorie", categories)
    providers_in_cat = [n for n, c in LLM_PROVIDERS.items() if c["category"] == selected_cat]

    if providers_in_cat:
        selected_provider = st.selectbox("Fournisseur", providers_in_cat)
        config = LLM_PROVIDERS[selected_provider]
        model = st.selectbox("Modèle", config["models"],
                             index=config["models"].index(config["default_model"]))

        if selected_provider == "Ollama (local)":
            available, models = check_ollama_available()
            if not available:
                st.error("⚠️ Ollama n'est pas lancé. Exécutez `ollama serve`.")
                st.code("""
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama serve
                """, language="bash")
            else:
                st.success(f"✅ Ollama disponible ({len(models)} modèles)")
                if models:
                    model = st.selectbox("Modèles installés", models)

        st.info(f"**Description** : {config['description']} | **Limite** : {config['rate_limit']}")

        if st.button("🔌 Tester la connexion", type="primary", use_container_width=True):
            with st.spinner("Test en cours..."):
                response = run_multi_llm_chat("Dis bonjour.", [], selected_provider,
                                              model_override=model, max_tokens=100)
                if response and not response.startswith(("❌", "⚠️")):
                    st.success("✅ Connexion réussie !")
                    st.markdown(f"**Réponse** : {response}")
                    st.session_state.llm_provider = selected_provider
                    st.session_state.llm_model = model
                else:
                    st.error(response)


elif page == "🤖 Copilote Multi-LLM":
    st.title("🤖 BioStruct Copilot (Multi-LLM)")
    col1, col2 = st.columns([2, 1])
    with col1:
        provider = st.selectbox("Fournisseur LLM", list(LLM_PROVIDERS.keys()),
                                index=list(LLM_PROVIDERS.keys()).index(st.session_state.llm_provider))
    with col2:
        config = LLM_PROVIDERS[provider]
        model = st.selectbox("Modèle", config["models"],
                             index=config["models"].index(config["default_model"]))

    client, error = get_llm_client(provider)
    if error:
        st.warning(f"⚠️ {error}")
        st.info("💡 Allez dans **🤖 Configuration LLM** pour configurer.")

    for msg in st.session_state.copilot_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input(f"Posez votre question à {provider}..."):
        st.session_state.copilot_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner(f"Réflexion via {provider}..."):
                tools = build_copilot_tools() if provider in ("OpenAI (payant)", "Groq (gratuit)") else None
                response = run_multi_llm_chat(prompt, st.session_state.copilot_history[:-1],
                                              provider, model_override=model, tools=tools)
                st.markdown(response)
        st.session_state.copilot_history.append({"role": "assistant", "content": response})

    if st.session_state.copilot_history:
        if st.button("🗑️ Effacer la conversation", use_container_width=True):
            st.session_state.copilot_history = []; st.rerun()


elif page == "🧬 Modèles spécialisés":
    st.title("🧬 Modèles Spécialisés en Biologie")
    rows = [{"Modèle": k, "Description": v["description"],
             "Spécialité": v["specialty"], "Taille": v["size"]}
            for k, v in SPECIALIZED_MODELS.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    model_name = st.selectbox("Modèle", list(SPECIALIZED_MODELS.keys()))
    info = SPECIALIZED_MODELS[model_name]
    st.info(f"**{model_name}** — {info['description']} | **Spécialité** : {info['specialty']}")

    prompt = st.text_area("Votre question",
        value="Explique le rôle de la protéine Tau dans la maladie d'Alzheimer.", height=100)

    if st.button("🚀 Interroger", type="primary", use_container_width=True):
        with st.spinner(f"Interrogation de {model_name}..."):
            st.markdown(query_specialized_model(model_name, prompt))


elif page == "🔬 Visualisation 3D":
    st.title("🔬 Visualisation 3D")
    col1, col2 = st.columns([1, 3])
    with col1:
        pdb_id = st.text_input("ID PDB", value="1A2C").strip()
        if st.button("📥 Charger", use_container_width=True):
            try:
                st.session_state.pdb_text = fetch_pdb_structure(pdb_id)
                st.session_state.structure = parse_pdb_structure(st.session_state.pdb_text)
                st.success("Chargée !")
            except Exception as e: st.error(f"Erreur : {e}")
        if st.session_state.structure:
            for k, v in get_structure_info(st.session_state.structure).items():
                st.write(f"**{k}:** {v}")
            style = st.selectbox("Style", ["cartoon", "stick", "sphere", "line"])
            color = st.selectbox("Couleur", ["chain", "residue", "spectrum"])
    with col2:
        if st.session_state.pdb_text:
            view = py3Dmol.view(width=800, height=600)
            view.addModel(st.session_state.pdb_text, "pdb")
            view.setStyle({style: {"color": color}})
            view.zoomTo(); showmol(view, height=600, width=800)


elif page == "✅ Validation":
    st.title("✅ Validation Automatique")
    if st.session_state.structure is None:
        st.warning("Chargez une structure.")
    else:
        if st.button("🚀 Valider", type="primary", use_container_width=True):
            st.session_state.validation_report = validate_structure(st.session_state.structure)
        if st.session_state.validation_report:
            r = st.session_state.validation_report
            c1, c2, c3 = st.columns(3)
            c1.metric("Score", f"{r['score_global']}/100")
            c2.metric("Outliers", r["outliers_ramachandran"])
            c3.metric("Clashes", r["clashes_steriques"])
            st.markdown(f"### {r['verdict']}")


elif page == "🧪 Docking":
    st.title("🧪 Docking")
    if st.session_state.pdb_text is None:
        st.warning("Chargez une structure.")
    else:
        if not PANDADOCK_AVAILABLE: st.info("ℹ️ Scores simulés.")
        cx = st.number_input("Centre X", value=0.0)
        cy = st.number_input("Centre Y", value=0.0)
        cz = st.number_input("Centre Z", value=0.0)
        smiles = st.text_area("SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O")
        if st.button("🚀 Docking", type="primary", use_container_width=True):
            st.session_state.docking_results = run_docking(
                st.session_state.pdb_text, smiles, (cx, cy, cz), (20.0, 20.0, 20.0))
        if st.session_state.docking_results:
            st.dataframe(pd.DataFrame(st.session_state.docking_results)[["Pose", "Score (kcal/mol)"]],
                         use_container_width=True)


elif page == "🎯 Docking aveugle":
    st.title("🎯 Docking Aveugle")
    if st.session_state.structure is None:
        st.warning("Chargez une structure.")
    else:
        if st.button("🔍 Détecter les poches", type="primary", use_container_width=True):
            st.session_state.cavities = detect_cavities(st.session_state.structure)
            st.success(f"✅ {len(st.session_state.cavities)} poches")
        if st.session_state.cavities:
            st.dataframe(pd.DataFrame([{"Poche": f"P{i+1}", "Volume (Å³)": c["volume_A3"],
                "Druggabilité": c.get("druggabilité")} for i, c in enumerate(st.session_state.cavities)]),
                use_container_width=True, hide_index=True)
            smiles = st.text_area("SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O", key="sb")
            if st.button("🚀 Docking aveugle", use_container_width=True):
                results = []
                for i, cav in enumerate(st.session_state.cavities):
                    box = max(15, min(30, (cav["volume_A3"] ** (1/3)) * 4))
                    dr = run_docking(st.session_state.pdb_text, smiles, cav["center"], (box, box, box))
                    best = min(d["Score (kcal/mol)"] for d in dr)
                    results.append({"Poche": f"P{i+1}", "Volume": cav["volume_A3"],
                                   "Meilleur score": round(best, 2)})
                st.session_state.blind_docking_results = sorted(results, key=lambda x: x["Meilleur score"])
                st.dataframe(pd.DataFrame(st.session_state.blind_docking_results),
                            use_container_width=True, hide_index=True)


elif page == "🔗 Docking PPI":
    st.title("🔗 Docking Protéine-Protéine")
    if st.session_state.structure is None:
        st.warning("Chargez la protéine A.")
    else:
        pdb_b = st.text_input("ID PDB protéine B", value="1BRS").strip().upper()
        if st.button("📥 Charger B", use_container_width=True):
            try:
                st.session_state.pdb_text_b = fetch_pdb_structure(pdb_b)
                st.session_state.structure_b = parse_pdb_structure(st.session_state.pdb_text_b)
                st.success("Protéine B chargée")
            except Exception as e: st.error(f"Erreur : {e}")
        if st.session_state.get("structure_b"):
            cutoff = st.slider("Distance contact (Å)", 3.0, 10.0, 6.0)
            if st.button("🔬 Analyser", type="primary", use_container_width=True):
                ia, ib = identify_interface_residues(st.session_state.structure,
                                                     st.session_state.structure_b, cutoff)
                st.session_state.ppi_results = {"A": ia, "B": ib, "nA": len(ia), "nB": len(ib)}
            if st.session_state.ppi_results:
                p = st.session_state.ppi_results
                c1, c2 = st.columns(2)
                c1.metric("Résidus A", p["nA"]); c2.metric("Résidus B", p["nB"])
                st.write("**A :**", ", ".join(p["A"][:30]))
                st.write("**B :**", ", ".join(p["B"][:30]))


elif page == "🧬 Morphométrie":
    st.title("🧬 Signature Morphométrique")
    if st.session_state.structure is None:
        st.warning("Chargez une structure.")
    else:
        label = st.text_input("Label", value=st.session_state.structure.header.get("idcode", "Protéine"))
        if st.button("🔬 Calculer", type="primary", use_container_width=True):
            sig = compute_signature(st.session_state.structure, label)
            if sig:
                st.session_state.last_signature = sig
                st.session_state.signatures_library[label] = sig
                st.success(f"✅ {label}")
        if st.session_state.last_signature:
            sig = st.session_state.last_signature
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rg", f"{sig.get('rayon_giration')} Å")
            m2.metric("Fractale", sig.get('dimension_fractale'))
            m3.metric("Compacité", sig.get('compacité'))
            m4.metric("Résidus", sig.get('nb_résidus'))
            st.dataframe(pd.DataFrame([{"Propriété": k, "Valeur": v}
                for k, v in sig.items() if isinstance(v, (int, float)) and not k.startswith("freq_")]),
                use_container_width=True, hide_index=True)


elif page == "🎯 Landmarks auto":
    st.title("🎯 Landmarks Automatiques")
    if st.session_state.structure is None:
        st.warning("Chargez une structure.")
    else:
        n = st.slider("Nombre", 5, 50, 20)
        if st.button("🔍 Détecter", type="primary", use_container_width=True):
            st.session_state.landmarks = detect_landmarks(st.session_state.structure, n)
        if st.session_state.landmarks:
            st.dataframe(pd.DataFrame(st.session_state.landmarks), use_container_width=True, hide_index=True)
            view = py3Dmol.view(width=800, height=500)
            view.addModel(st.session_state.pdb_text, "pdb")
            view.setStyle({"cartoon": {"color": "lightgray"}})
            for lm in st.session_state.landmarks:
                view.addSphere({"center": {"x": lm["x"], "y": lm["y"], "z": lm["z"]},
                                "radius": 0.8, "color": "red", "opacity": 0.8})
            view.zoomTo(); showmol(view, height=500, width=800)


elif page == "🔗 Morpho-topologie":
    st.title("🔗 Indice Morpho-Topologique")
    if st.session_state.last_signature is None:
        st.warning("Calculez une signature.")
    else:
        idx = compute_morpho_topology_index(st.session_state.last_signature)
        c1, c2 = st.columns(2)
        c1.metric("Indice", idx["indice_morpho_topologique"])
        c2.metric("Classe", idx["classification_morpho_topo"])
        c1, c2 = st.columns(2)
        c1.metric("Axe Forme", idx["forme_axe"]); c2.metric("Axe Topologie", idx["topologie_axe"])


elif page == "💊 Druggabilité":
    st.title("💊 Indice de Druggabilité")
    if st.session_state.last_signature is None:
        st.warning("Calculez une signature.")
    else:
        idx = compute_druggability_index(st.session_state.last_signature)
        c1, c2 = st.columns(2)
        c1.metric("Indice", idx["indice_druggabilité"])
        c2.metric("Classification", idx["classification_druggabilité"])
        comp = pd.DataFrame([{"Composante": k.replace("score_", ""), "Score": v}
            for k, v in idx.items() if k.startswith("score_")])
        st.dataframe(comp, use_container_width=True, hide_index=True)


elif page == "🌍 Variants génétiques":
    st.title("🌍 Variants Génétiques")
    rows = []
    for gene, data in POPULATION_VARIANTS_DB.items():
        for rsid, alleles in data.items():
            for allele, freqs in alleles.items():
                rows.append({"Gène": gene, "rsID": rsid, "Allèle": allele,
                            "Afrique": freqs.get("Afrique", 0),
                            "Europe": freqs.get("Europe", 0),
                            "Asie": freqs.get("Asie", 0)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


elif page == "🌡️ Impact environnemental":
    st.title("🌡️ Impact Environnemental")
    if st.session_state.last_signature is None:
        st.warning("Calculez une signature.")
    else:
        ph = st.slider("pH", 3.0, 11.0, 7.4, 0.1)
        temp = st.slider("Température (°C)", 0, 50, 37)
        salt = st.slider("Sel (mM)", 0, 1000, 150)
        alt = st.slider("Altitude (m)", 0, 5000, 0, 100)
        sig = st.session_state.last_signature
        stab = 70 + 20 * sig.get("compacité", 0.5) - abs(sig.get("gravité_hydrophobicité", 0)) * 2
        stab -= abs(ph - 7.4) * 0.1
        if temp > 40: stab -= 0.5 * (temp - 40) / 10
        if salt > 500: stab -= 0.3 * (salt - 500) / 500
        if alt > 2000: stab -= 0.2 * (alt - 2000) / 3000
        st.metric("Stabilité globale", f"{max(0, min(100, stab)):.1f}/100")


elif page == "🌐 Comparaison populations":
    st.title("🌐 Comparaison Inter-Populations")
    c1, c2 = st.columns(2)
    with c1: pa = st.selectbox("Population A", ["Afrique", "Europe", "Asie"], index=0)
    with c2: pb = st.selectbox("Population B", ["Afrique", "Europe", "Asie"], index=1)
    if pa != pb:
        rows = []
        for gene, data in POPULATION_VARIANTS_DB.items():
            for rsid, alleles in data.items():
                for allele, freqs in alleles.items():
                    diff = abs(freqs.get(pa, 0) - freqs.get(pb, 0)) * 100
                    rows.append({"Gène": gene, "rsID": rsid, "Allèle": allele,
                                f"Fréq {pa}": freqs.get(pa, 0), f"Fréq {pb}": freqs.get(pb, 0),
                                "Différence (%)": round(diff, 2)})
        st.dataframe(pd.DataFrame(rows).sort_values("Différence (%)", ascending=False),
                    use_container_width=True, hide_index=True)


elif page == "🔬 Découverte biomarqueurs":
    st.title("🔬 Découverte de Biomarqueurs")
    if len(st.session_state.signatures_library) < 4:
        st.warning("⚠️ Il faut au moins 4 signatures.")
    else:
        labels = list(st.session_state.signatures_library.keys())
        groups_det = list(set([l.split("_")[0] for l in labels]))
        c1, c2 = st.columns(2)
        with c1: ga = st.selectbox("Groupe A", groups_det, index=0)
        with c2: gb = st.selectbox("Groupe B", groups_det, index=min(1, len(groups_det)-1))
        if ga != gb:
            df_data = []
            for lab, sig in st.session_state.signatures_library.items():
                grp = lab.split("_")[0]
                if grp not in (ga, gb): continue
                row = {"label": lab, "_group": grp}
                for k, v in sig.items():
                    if isinstance(v, (int, float)): row[k] = v
                df_data.append(row)
            df = pd.DataFrame(df_data)
            feature_cols = [c for c in df.columns if c not in ("label", "_group")]
            if st.button("🔬 Analyser", type="primary", use_container_width=True):
                df_stats = statistical_test_features(df, [ga, gb], feature_cols)
                st.session_state.biomarker_results = {"df_ranked": rank_biomarkers(df_stats)}
            if st.session_state.biomarker_results:
                st.dataframe(st.session_state.biomarker_results["df_ranked"].head(10),
                            use_container_width=True, hide_index=True)


elif page == "📊 Transcriptomique":
    st.title("📊 Analyse Transcriptomique")
    if st.button("🎲 Générer dataset simulé", use_container_width=True):
        st.session_state.expression_matrix, st.session_state.expression_groups = simulate_expression_matrix()
        st.success("Dataset généré")
    if st.session_state.expression_matrix is not None:
        if st.button("🚀 Analyse différentielle", type="primary", use_container_width=True):
            st.session_state.de_results = differential_expression_analysis(
                st.session_state.expression_matrix, st.session_state.expression_groups)
        if st.session_state.de_results is not None:
            st.metric("Gènes DE", st.session_state.de_results["DE significatif"].sum())
            st.dataframe(st.session_state.de_results.head(20), use_container_width=True, hide_index=True)


elif page == "📈 Analyse de survie":
    st.title("📈 Analyse de Survie")
    if not LIFELINES_AVAILABLE:
        st.error("lifelines non installé.")
    else:
        uploaded = st.file_uploader("CSV survie (colonnes T, E)", type=["csv"])
        if uploaded:
            st.session_state.survival_data = pd.read_csv(uploaded)
        if st.session_state.survival_data is not None:
            df = st.session_state.survival_data
            if "T" in df.columns and "E" in df.columns:
                if st.button("📊 Kaplan-Meier", type="primary", use_container_width=True):
                    kmf = KaplanMeierFitter()
                    kmf.fit(df["T"], event_observed=df["E"], label="Tous")
                    st.session_state.survival_result = kmf
                if st.session_state.get("survival_result"):
                    st.write(f"Survie médiane : {st.session_state.survival_result.median_survival_time_}")


elif page == "🤖 Prédiction ML":
    st.title("🤖 Prédiction ML (XGBoost)")
    if st.session_state.de_results is None:
        st.warning("Lancez d'abord l'analyse transcriptomique.")
    elif not XGBOOST_AVAILABLE:
        st.error("XGBoost non installé.")
    else:
        top_n = st.slider("Top gènes", 20, 200, 50)
        if st.button("🚀 Entraîner", type="primary", use_container_width=True):
            df_ml = prepare_ml_data(st.session_state.de_results)
            st.session_state.ml_result = train_xgboost_classifier(df_ml, top_n)
        if st.session_state.ml_result:
            m = st.session_state.ml_result["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", m["Accuracy"]); c2.metric("AUC", m["AUC"])
            c3.metric("Precision", m["Precision"]); c4.metric("Recall", m["Recall"])
            st.dataframe(st.session_state.ml_result["importance"], use_container_width=True, hide_index=True)


elif page == "📊 GEPIA2 (TCGA)":
    st.title("📊 GEPIA2 — Expression TCGA")
    gene = st.text_input("Gène", value="TP53").strip().upper()
    cancer = st.selectbox("Cancer", list(UALCAN_CANCERS.keys()))
    if st.button("🔍 Analyser", type="primary", use_container_width=True):
        df = simulate_gepia2_expression(gene)
        tumor = df[df["Type"] == "Tumeur"]["Expression"]
        normal = df[df["Type"] == "Normal"]["Expression"]
        _, p = ttest_ind(tumor, normal, equal_var=False)
        fc = tumor.mean() / normal.mean() if normal.mean() > 0 else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Tumeur", round(tumor.mean(), 2))
        c2.metric("Normal", round(normal.mean(), 2))
        c3.metric("FC", round(fc, 2))
        st.metric("p-value", f"{p:.2e}")


elif page == "🧪 UALCAN (sous-groupes)":
    st.title("🧪 UALCAN")
    gene = st.text_input("Gène", value="TP53").strip().upper()
    cancer = st.selectbox("Cancer", list(UALCAN_CANCERS.keys()))
    sub = st.selectbox("Sous-groupe", ["stage", "grade", "age", "gender", "race"])
    if st.button("🔍 Analyser", type="primary", use_container_width=True):
        df = simulate_ualcan_subgroups(gene, cancer, sub)
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)


elif page == "🧪 Mutagenèse in silico":
    st.title("🧪 Mutagenèse In Silico")
    if st.session_state.structure is None:
        st.warning("Chargez une structure.")
    else:
        n_mut = st.number_input("Nombre de mutations", 1, 10, 1)
        mutations = []
        for i in range(n_mut):
            c1, c2, c3 = st.columns(3)
            with c1: chain = st.text_input(f"Chaîne {i+1}", value="A", key=f"m_ch_{i}")
            with c2: pos = st.number_input(f"Position {i+1}", value=1, key=f"m_pos_{i}")
            with c3: new_aa = st.selectbox(f"→ AA", list(AA_PROPERTIES.keys()), key=f"m_aa_{i}")
            try: orig = st.session_state.structure[0][chain][(" ", pos, " ")].resname
            except Exception: orig = "ALA"
            mutations.append({"Chaîne": chain, "Position": pos, "Original": orig, "Mutant": new_aa})
        if st.button("🔬 Analyser", type="primary", use_container_width=True):
            results = []
            for mut in mutations:
                imp = predict_mutation_impact(st.session_state.structure, mut)
                if imp:
                    results.append({"Mutation": imp["mutation"],
                                   "ΔΔG (kcal/mol)": imp["ΔΔG_prédit"],
                                   "Stabilité": imp["stabilité"]})
            st.session_state.mutation_results = pd.DataFrame(results)
        if st.session_state.mutation_results is not None:
            st.dataframe(st.session_state.mutation_results, use_container_width=True, hide_index=True)


elif page == "🔧 Saturation mutagenèse":
    st.title("🔧 Saturation Mutagenèse")
    if st.session_state.structure is None:
        st.warning("Chargez une structure.")
    else:
        chains = [c.id for c in st.session_state.structure.get_chains()]
        c1, c2 = st.columns(2)
        with c1: chain = st.selectbox("Chaîne", chains)
        with c2:
            try: positions = [r.id[1] for r in st.session_state.structure[0][chain].get_residues() if r.id[0] == " "]
            except Exception: positions = [1]
            pos = st.selectbox("Position", positions)
        if st.button("🔬 Générer les 19 variants", type="primary", use_container_width=True):
            st.session_state.saturation_results = saturation_mutagenesis(st.session_state.structure, chain, pos)
        if st.session_state.saturation_results is not None and len(st.session_state.saturation_results) > 0:
            st.dataframe(st.session_state.saturation_results, use_container_width=True, hide_index=True)


elif page == "🛡️ Stabilisation (PROSS-like)":
    st.title("🛡️ Stabilisation de Protéine")
    if st.session_state.structure is None:
        st.warning("Chargez une structure.")
    else:
        if st.button("🔬 Proposer des mutations", type="primary", use_container_width=True):
            st.session_state.stabilization_results = propose_stabilizing_mutations(st.session_state.structure)
        if st.session_state.stabilization_results is not None and len(st.session_state.stabilization_results) > 0:
            st.dataframe(st.session_state.stabilization_results, use_container_width=True, hide_index=True)


elif page == "🔗 Ponts disulfures":
    st.title("🔗 Ponts Disulfures")
    if st.session_state.structure is None:
        st.warning("Chargez une structure.")
    else:
        if st.button("🔍 Analyser", type="primary", use_container_width=True):
            st.session_state.disulfide_results = identify_disulfide_opportunities(st.session_state.structure)
        if st.session_state.disulfide_results is not None and len(st.session_state.disulfide_results) > 0:
            st.dataframe(st.session_state.disulfide_results, use_container_width=True, hide_index=True)


elif page == "📚 Bibliothèque de variants":
    st.title("📚 Bibliothèque de Variants")
    if st.session_state.structure is None:
        st.warning("Chargez une structure.")
    else:
        positions_str = st.text_input("Positions (virgule)", value="10, 15, 20")
        chain = st.text_input("Chaîne", value="A")
        try: positions = [int(p.strip()) for p in positions_str.split(",") if p.strip()]
        except Exception: positions = []
        if positions:
            if st.button("🎨 Générer", type="primary", use_container_width=True):
                with st.spinner("Génération..."):
                    st.session_state.variant_library = generate_variant_library(
                        st.session_state.structure, [(chain, p) for p in positions])
            if st.session_state.get("variant_library") is not None:
                df = st.session_state.variant_library
                st.success(f"✅ {len(df)} variants")
                st.dataframe(df.head(50), use_container_width=True, hide_index=True)
                st.download_button("📥 CSV", df.to_csv(index=False).encode(),
                                   "variant_library.csv", "text/csv")


elif page == "🎨 Visualisation 3D mutations":
    st.title("🎨 Visualisation 3D des Mutations")
    if st.session_state.structure is None:
        st.warning("Chargez une structure.")
    else:
        ddg_data = []
        if st.session_state.mutation_results is not None:
            ddg_data = [{"Mutation": r["Mutation"], "ΔΔG": r["ΔΔG (kcal/mol)"],
                        "Position": r["Mutation"]} for _, r in st.session_state.mutation_results.iterrows()]
        elif st.session_state.get("variant_library") is not None:
            df = st.session_state.variant_library.head(30)
            ddg_data = [{"Mutation": r["Variant_ID"], "ΔΔG": r["ΔΔG (kcal/mol)"]}
                        for _, r in df.iterrows()]
        if not ddg_data:
            st.warning("Générez d'abord des mutations.")
        else:
            view = py3Dmol.view(width=800, height=600)
            view.addModel(st.session_state.pdb_text, "pdb")
            view.setStyle({"cartoon": {"color": "lightgray"}})
            for entry in ddg_data[:50]:
                try:
                    pos = int(''.join(filter(str.isdigit, entry["Mutation"])))
                    ddg = entry["ΔΔG"]
                    color = "green" if ddg < -0.5 else "red" if ddg > 0.5 else "yellow"
                    view.addStyle({"resi": pos}, {"stick": {"color": color}})
                except Exception: continue
            view.zoomTo(); showmol(view, height=600, width=800)
            st.caption("🟢 Vert = stabilisant | 🟡 Jaune = neutre | 🔴 Rouge = déstabilisant")


elif page == "📋 Protocole laboratoire":
    st.title("📋 Protocole de Laboratoire")
    if st.session_state.structure is None:
        st.warning("Chargez une structure.")
    elif st.session_state.get("mutation_results") is None:
        st.warning("Lancez d'abord une mutagenèse in silico.")
    else:
        df = st.session_state.mutation_results
        mutations = []
        for _, row in df.iterrows():
            mut = row["Mutation"]
            try:
                pos = int(''.join(filter(str.isdigit, mut)))
                mutations.append({"Original": mut[:3], "Position": pos, "Mutant": mut[-3:]})
            except Exception: continue
        if mutations:
            enzyme = st.selectbox("Enzyme", ["Q5", "PfuUltra", "Phusion", "KOD"])
            if st.button("📋 Générer le protocole", type="primary", use_container_width=True):
                protocol, primers = generate_lab_protocol(st.session_state.structure, mutations, enzyme)
                st.session_state.lab_protocol = protocol
                st.session_state.lab_primers = primers
            if st.session_state.lab_protocol:
                st.markdown(st.session_state.lab_protocol)
                st.download_button("📥 Télécharger", st.session_state.lab_protocol.encode(),
                                   "protocole.txt", "text/plain")


elif page == "📊 Mes Analyses":
    st.title("📊 Historique de mes Analyses")
    if not st.session_state.user or st.session_state.user.get("offline"):
        st.warning("Connectez-vous avec Supabase configuré.")
    else:
        analyses = load_analyses(st.session_state.user["id"])
        if not analyses:
            st.info("Aucune analyse.")
        for a in analyses:
            with st.expander(f"🔬 {a['analysis_type']} — {a['created_at'][:19]}"):
                st.json(a["input_data"]); st.json(a["results"])


# --- PIED DE PAGE ---
st.markdown("---")
st.caption("🧬 **BioStruct AI v14.0** — Multi-LLM (Cloud gratuit, Local, Spécialisé)")
