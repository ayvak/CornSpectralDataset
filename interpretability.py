import shap

def compute_shap_values(model, X_sample):
    explainer = shap.KernelExplainer(model.predict, X_sample[:100]) 
    # For large data, sample a subset
    shap_values = explainer.shap_values(X_sample[:100])
    return shap_values

def plot_shap_summary(shap_values, X_sample):
    shap.summary_plot(shap_values, X_sample[:100])