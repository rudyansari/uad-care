# app.py - Streamlit Interactive Application for UAD-CARE (Unified Anomaly Detection with Counterfactual Actionable Reasoning & Explanation)
# Run with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch
import torch.nn as nn
import shap
from scipy.optimize import minimize
from scipy.linalg import expm
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="UAD-CARE Advisor", page_icon="🎓", layout="wide")

# ============================================================
# 1. Define VAE Model (same as training)
# ============================================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

def train_vae(X_normal, input_dim, epochs=30, lr=1e-3, batch_size=64):
    model = VAE(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_normal, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            recon, mean, logvar = model(x)
            recon_loss = nn.MSELoss()(recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            loss = recon_loss + 0.001 * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"VAE Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    return model

# ============================================================
# 2. Hybrid Risk Score Function
# ============================================================
def hybrid_risk_score(x, if_model, vae_model, xgb_model, alpha=0.2, beta=0.3, gamma=0.5):
    if_score = if_model.score_samples(x.reshape(1, -1))[0]
    if_score_norm = np.clip((if_score + 0.5) / 0.5, 0, 1)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    recon, _, _ = vae_model(x_tensor)
    vae_score = nn.MSELoss()(recon, x_tensor).item()
    vae_score_norm = np.clip(vae_score / 10.0, 0, 1)
    xgb_score = xgb_model.predict_proba(x.reshape(1, -1))[0][0]
    risk = alpha * if_score_norm + beta * vae_score_norm + gamma * xgb_score
    return risk

# ============================================================
# 3. Standalone NOTEARS (causal graph)
# ============================================================
def notears_linear(X, lambda1=0.05, max_iter=100, h_tol=1e-8, rho_max=1e16, w_threshold=0.05):
    n, d = X.shape
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    
    def _loss(W_vec):
        W_mat = W_vec.reshape(d, d)
        obj = 0.5 / n * np.linalg.norm(X_std - X_std @ W_mat, 'fro') ** 2
        obj += lambda1 * np.sum(np.abs(W_mat))
        return obj
    
    def _grad(W_vec):
        W_mat = W_vec.reshape(d, d)
        grad_mat = -1.0 / n * X_std.T @ (X_std - X_std @ W_mat)
        grad_mat += lambda1 * np.sign(W_mat)
        return grad_mat.flatten()
    
    def _h(W_vec):
        W_mat = W_vec.reshape(d, d)
        M = W_mat * W_mat
        h_val = np.trace(expm(M)) - d
        return h_val
    
    def _grad_h(W_vec):
        W_mat = W_vec.reshape(d, d)
        M = W_mat * W_mat
        exp_M = expm(M)
        grad_h_mat = 2 * W_mat * exp_M.T
        return grad_h_mat.flatten()
    
    W_est = np.zeros(d * d)
    rho = 1.0
    alpha = 0.0
    
    for it in range(max_iter):
        def aug_loss(W):
            return _loss(W) + alpha * _h(W) + 0.5 * rho * _h(W) ** 2
        def aug_grad(W):
            return _grad(W) + (alpha + rho * _h(W)) * _grad_h(W)
        from scipy.optimize import minimize
        res = minimize(aug_loss, W_est, method='L-BFGS-B', jac=aug_grad,
                       options={'maxiter': 100, 'disp': False})
        W_est = res.x
        h_val = _h(W_est)
        if h_val > 0.25 * h_tol:
            alpha += rho * h_val
            rho = min(rho * 10, rho_max)
        else:
            if h_val <= h_tol:
                break
    W = W_est.reshape(d, d)
    W[np.abs(W) < w_threshold] = 0
    np.fill_diagonal(W, 0)
    return W

# ============================================================
# 4. Counterfactual generation
# ============================================================
def generate_counterfactual(x, risk_func, if_model, vae_model, xgb_model,
                            feature_names, W, theta=0.7, step_size=0.1):
    x_current = x.copy()
    risk_current = risk_func(x_current, if_model, vae_model, xgb_model)
    if risk_current < theta:
        return x_current, None, risk_current
    grad = np.zeros_like(x_current)
    for j in range(len(x_current)):
        x_plus = x_current.copy()
        delta = step_size * (np.std(x_current) if np.std(x_current) > 0 else 1.0)
        x_plus[j] += delta
        risk_plus = risk_func(x_plus, if_model, vae_model, xgb_model)
        grad[j] = (risk_plus - risk_current) / delta
    sorted_idx = np.argsort(np.abs(grad))[::-1]
    for idx in sorted_idx:
        x_candidate = x_current.copy()
        delta = step_size * (np.std(x_current) if np.std(x_current) > 0 else 1.0)
        x_candidate[idx] += delta
        risk_new = risk_func(x_candidate, if_model, vae_model, xgb_model)
        if risk_new < theta:
            return x_candidate, idx, risk_new
    return None, None, risk_current

# ============================================================
# 5. Load and train models (cached)
# ============================================================
@st.cache_resource
def load_and_train():
    # Load dataset
    df = pd.read_csv('umbjm.csv', sep=';')
    df.rename(columns={
        'IPS Semester 1': 'Semester 1 GPA',
        'IPS Semester 2': 'Semester 2 GPA',
        'Jumlah sks Semester Semester 1': 'Semester 1 Credits',
        'Jumlah sks Semester Semester 2': 'Semester 2 Credits',
        'Tahun_PMB': 'Admission Year'
    }, inplace=True)
    features = ['Semester 1 GPA', 'Semester 2 GPA', 'Semester 1 Credits', 'Semester 2 Credits', 'Admission Year']
    df = df[features + ['Target']].dropna()
    X = df[features].values
    y = df['Target'].values
    feature_names = features
    
    # Train/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Isolation Forest
    if_model = IsolationForest(contamination=0.25, random_state=42)
    if_model.fit(X_train)
    
    # VAE
    X_normal = X_train[y_train == 1]
    if len(X_normal) < 10:
        X_normal = X_train
    vae_model = train_vae(X_normal, input_dim=X.shape[1], epochs=30)
    
    # XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    # Causal graph
    W = notears_linear(X_train, lambda1=0.05, w_threshold=0.05)
    
    # SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    
    return if_model, vae_model, xgb_model, W, feature_names, explainer, X, y

# ============================================================
# 6. Streamlit UI
# ============================================================
st.title("🎓 UAD-CARE: Unified Anomaly Detection with Counterfactual Reasoning and Explanation")
st.markdown("""
**U**nified **A**nomaly **D**etection with **C**ounterfactual **A**ctionable **R**easoning & **E**xplanation

This application uses **UAD-CARE**, an anomaly detection framework that combines **hybrid anomaly scoring**, **causal graph discovery**, and **counterfactual reasoning** to provide actionable intervention recommendations for students at risk of non‑graduation.
""")

# Load models (once)
with st.spinner("Loading models and data (first time only, please wait)..."):
    if_model, vae_model, xgb_model, W, feature_names, explainer, X_all, y_all = load_and_train()

# Sidebar input
st.sidebar.header("📝 Student Data")
input_data = {}
for feat in feature_names:
    if feat in ['Semester 1 GPA', 'Semester 2 GPA']:
        val = st.sidebar.number_input(feat, min_value=0.0, max_value=4.0, value=2.5, step=0.01)
    elif feat in ['Semester 1 Credits', 'Semester 2 Credits']:
        val = st.sidebar.number_input(feat, min_value=0, max_value=30, value=20, step=1)
    else:  # Admission Year
        val = st.sidebar.selectbox(feat, [2019, 2020, 2021, 2022, 2023], index=2)
    input_data[feat] = val

# Tombol prediksi
if st.sidebar.button("🔍 Analyze Risk"):
    x_input = np.array([input_data[feat] for feat in feature_names]).astype(float)
    
    # Hitung risiko
    risk = hybrid_risk_score(x_input, if_model, vae_model, xgb_model)
    
    # Tampilkan skor risiko dengan warna
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Risk Score", f"{risk:.1%}", delta="⚠️ High Risk" if risk > 0.7 else "✅ Normal")
        if risk > 0.7:
            st.error("⚠️ This student is at HIGH RISK of non‑graduation.")
        else:
            st.success("✅ This student is LIKELY to graduate on time.")
    
    # Counterfactual
    def risk_func_wrapper(x, if_m, vae_m, xgb_m):
        return hybrid_risk_score(x, if_m, vae_m, xgb_m)
    
    x_cf, cf_idx, new_risk = generate_counterfactual(
        x_input, risk_func_wrapper, if_model, vae_model, xgb_model,
        feature_names, W, theta=0.7, step_size=0.1
    )
    
    with col2:
        if x_cf is not None and cf_idx is not None:
            cf_feature = feature_names[cf_idx]
            old_val = x_input[cf_idx]
            new_val = x_cf[cf_idx]
            reduction = risk - new_risk
            st.info(f"💡 **Minimal Counterfactual Intervention**")
            st.write(f"Change **{cf_feature}** from **{old_val:.2f}** to **{new_val:.2f}**")
            st.write(f"📉 Risk would drop from **{risk:.1%}** to **{new_risk:.1%}** (reduction **{reduction:.1%}**).")
        else:
            st.warning("⚠️ No minimal counterfactual found within search budget.")
    
    # Penjelasan LLM (template)
    st.subheader("📝 LLM Explanation")
    # Get top features from SHAP for this sample
    shap_values = explainer.shap_values(x_input.reshape(1, -1))
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_abs = np.abs(shap_values).flatten()
    top_indices = np.argsort(shap_abs)[-3:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    if x_cf is not None and cf_idx is not None:
        cf_info = {'feature': feature_names[cf_idx], 'old_value': x_input[cf_idx], 'new_value': x_cf[cf_idx]}
    else:
        cf_info = {'feature': 'none', 'old_value': 0, 'new_value': 0}
    explanation = f"Student risk score: {risk:.2f}. Key factors: {', '.join(top_features)}. Recommended intervention: change {cf_info['feature']} from {cf_info['old_value']:.2f} to {cf_info['new_value']:.2f}."
    st.success(explanation)
    
    # SHAP force plot (optional)
    st.subheader("📊 SHAP Feature Contribution")
    # Create a simple bar chart of SHAP values
    fig, ax = plt.subplots()
    ax.barh(feature_names, shap_abs, color='steelblue')
    ax.set_xlabel('|SHAP Value|')
    ax.set_title('Feature Importance for this Student')
    st.pyplot(fig)
    
    # Historical context: show similar students
    st.subheader("📈 Students with Similar Profile")
    # Compute distances to all training samples (simple Euclidean)
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(x_input.reshape(1, -1), X_all).flatten()
    nearest_idx = np.argsort(distances)[1:6]  # 5 nearest excluding self (if self in training)
    nearest_targets = y_all[nearest_idx]
    grad_counts = (nearest_targets == 1).sum()
    dropout_counts = (nearest_targets == 0).sum()
    st.write(f"Among the 5 most similar students, **{grad_counts}** graduated and **{dropout_counts}** dropped out.")
    
    st.caption("UAD-CARE v1.0 | Model trained on Indonesian student data.")

else:
    st.info("👈 Enter student data in the sidebar, then click 'Analyze Risk'.")
    st.image("https://via.placeholder.com/800x200?text=UAD-CARE+Interactive+Dashboard", use_container_width=False)