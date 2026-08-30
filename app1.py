# -*- coding: utf-8 -*-
"""
TP Virtuel CPGE - Simulateur de Titrage Acido-Basique (Version Streamlit)
Préparé par Pr. ZAKARIA AIT EL CAID
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuration de la page Streamlit
st.set_page_config(page_title="TP Virtuel CPGE - Titrage Acido-Basique", layout="wide")

# Bandeau institutionnel supérieur
st.markdown("""
    <div style="background-color: #1a365d; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0; font-size: 20px;">🌟 SCRIPT PRÉPARÉ PAR LE PROFESSEUR : ZAKARIA AIT EL CAID 🌟</h2>
        <p style="color: #90cdf4; margin: 5px 0 0 0; font-size: 14px; font-style: italic;">
            Professor of Physical Sciences (PCSI / PSI) | Researcher in Corrosion and Materials Science | Casablanca, Morocco | +212 6 79 89 54 14
        </p>
    </div>
""", unsafe_allow_html=True)

st.title("🧪 Simulateur de Titrage Acido-Basique (pH-métrie & Conductimétrie)")

# Bibliothèque des acides (pKa stockés sous forme de tuple pour la stabilité du cache)
BIBLIOTHEQUE_ACIDES = {
    "Acide Chlorhydrique (Fort)": (-2.0,),
    "Acide Nitrique (Fort)": (-1.5,),
    "Acide Acétique (pKa = 4.76)": (4.76,),
    "Acide Formique (pKa = 3.75)": (3.75,),
    "Acide Benzoïque (pKa = 4.20)": (4.20,),
    "Acide Propanoïque (pKa = 4.87)": (4.87,),
    "Acide Hypochloreux (pKa = 7.53)": (7.53,),
    "Acide Oxalique (pKa = 1.25, 4.14)": (1.25, 4.14),
    "Acide Tartrique (pKa = 2.98, 4.34)": (2.98, 4.34),
    "Acide Carbonique (pKa = 6.35, 10.33)": (6.35, 10.33),
    "Acide Phosphorique (pKa = 2.15, 7.20, 12.35)": (2.15, 7.20, 12.35),
    "Acide Citrique (pKa = 3.13, 4.76, 6.40)": (3.13, 4.76, 6.40),
}

# --- BARRE LATÉRALE DE CONTRÔLE ---
st.sidebar.header("Paramètres du Titrage")
nom_acide = st.sidebar.selectbox("Choix de l'acide :", list(BIBLIOTHEQUE_ACIDES.keys()), index=2)
Ca = st.sidebar.slider("Concentration Ca (mol/L)", 0.001, 0.1, 0.01, 0.001)
Va = st.sidebar.slider("Volume initial Va (mL)", 5.0, 30.0, 20.0, 1.0)
Cb = st.sidebar.slider("Concentration Cb (mol/L)", 0.005, 0.2, 0.05, 0.005)
afficher_deriv = st.sidebar.checkbox("Afficher la dérivée dpH/dV")

pKa_tuple = BIBLIOTHEQUE_ACIDES[nom_acide]

# --- MOTEUR DE SIMULATION (OPTIMISÉ AVEC CACHE) ---
@st.cache_data
def simuler(pKa_vals_tuple, C_a, V_a_mL, C_b):
    pKa_list = np.sort(np.atleast_1d(pKa_vals_tuple))
    n = len(pKa_list)
    Ka = 10.0 ** (-pKa_list)
    Ke = 1.0e-14

    V_a = V_a_mL * 1e-3
    n_a_tot = C_a * V_a
    V_b_vals = np.linspace(0, 30.0, 450)

    pH_vals = np.zeros(len(V_b_vals))
    sigma_corr_vals = np.zeros(len(V_b_vals))

    lam_H, lam_OH, lam_Na, lam_anion = 350.0, 198.0, 50.1, 40.0

    for i, V_b_mL in enumerate(V_b_vals):
        V_b = V_b_mL * 1e-3
        V_tot = V_a + V_b
        C_0 = n_a_tot / V_tot
        C_Na = (C_b * V_b) / V_tot

        def f_pH(pH):
            H = 10.0 ** (-pH)
            OH = Ke / H
            denom = H ** n
            prod_Ka = 1.0
            for k in range(n):
                prod_Ka *= Ka[k]
                denom += prod_Ka * (H ** (n - 1 - k))

            somme_charges = 0.0
            prod_b = 1.0
            for j in range(1, n + 1):
                prod_b *= Ka[j - 1]
                somme_charges += j * prod_b * (H ** (n - j))

            A_charge = C_0 * (somme_charges / denom)
            return C_Na + H - OH - A_charge

        p_low, p_high = 0.0, 14.0
        for _ in range(60):
            p_mid = (p_low + p_high) / 2.0
            if f_pH(p_mid) > 0:
                p_low = p_mid
            else:
                p_high = p_mid

        pH_sol = (p_low + p_high) / 2.0
        pH_vals[i] = pH_sol

        H_eq = 10.0 ** (-pH_sol)
        OH_eq = Ke / H_eq

        denom = H_eq ** n
        prod_Ka = 1.0
        coeffs = [1.0]
        for k in range(n):
            prod_Ka *= Ka[k]
            coeffs.append(prod_Ka)
            denom += prod_Ka * (H_eq ** (n - 1 - k))

        alphas = [(coeffs[j] * (H_eq ** (n - j))) / denom for j in range(n + 1)]
        sigma_brute = lam_H * H_eq + lam_OH * OH_eq + lam_Na * C_Na
        for j in range(1, n + 1):
            sigma_brute += (j * lam_anion) * (C_0 * alphas[j])

        sigma_brute = sigma_brute / 10.0
        sigma_corr_vals[i] = sigma_brute * (V_tot / V_a)

    dpH_dV = np.gradient(pH_vals, V_b_vals)
    return V_b_vals, pH_vals, sigma_corr_vals, dpH_dV

V_b, pH_v, sig_v, dpH_v = simuler(pKa_tuple, Ca, Va, Cb)

# --- AFFICHAGE GRAPHIQUE ---
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.25)

ax_pH = fig.add_subplot(gs[0, 0])
ax_pH.plot(V_b, pH_v, color="#1f77b4", lw=2)
ax_pH.set_title("pH = f(Vb)", fontsize=10, fontweight="bold")
ax_pH.set_ylim(0, 14)
ax_pH.set_xlim(0, 30)
ax_pH.grid(True, alpha=0.3)
ax_pH.set_ylabel("pH")

ax_sigma = fig.add_subplot(gs[0, 1])
ax_sigma.plot(V_b, sig_v, color="#2ca02c", lw=2)
ax_sigma.set_title(r"$\sigma_{corr} = f(Vb)$", fontsize=10, fontweight="bold")
ax_sigma.set_xlim(0, 30)
ax_sigma.grid(True, alpha=0.3)
ax_sigma.set_ylabel(r"$\sigma_{corr}$ (mS/cm)")

ax_comb = fig.add_subplot(gs[1, :])
ax_comb.plot(V_b, pH_v, color="#1f77b4", lw=2.5, label="pH")

ax_comb_sigma = ax_comb.twinx()
ax_comb_sigma.plot(V_b, sig_v, color="#2ca02c", lw=2.5, linestyle="--", label=r"$\sigma_{corr}$")

if afficher_deriv:
    ax_comb_deriv = ax_comb.twinx()
    ax_comb_deriv.spines["right"].set_position(("outward", 60))
    ax_comb_deriv.plot(V_b, dpH_v, color="#d62728", lw=1.5, linestyle=":", label="dpH/dV")
    ax_comb_deriv.set_ylabel("dpH/dV", color="#d62728")
    ax_comb_deriv.tick_params(axis="y", labelcolor="#d62728")

ax_comb.set_title("Superposition pH & $\sigma_{corr}$", fontsize=10, fontweight="bold")
ax_comb.set_ylim(0, 14)
ax_comb.set_xlim(0, 30)
ax_comb.grid(True, alpha=0.3)
ax_comb.set_xlabel("Volume de NaOH versé Vb (mL)", fontweight="bold")
ax_comb.set_ylabel("pH", color="#1f77b4")
ax_comb_sigma.set_ylabel(r"$\sigma_{corr}$ (mS/cm)", color="#2ca02c")

st.pyplot(fig)