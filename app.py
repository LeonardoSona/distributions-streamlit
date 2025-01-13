import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, binom

# Streamlit App Title
st.title("Interactive Probability Distributions Explorer")
st.markdown("Explore **Discrete** and **Continuous** Probability Distributions with customizable parameters and dynamic visualizations.")

# Sidebar for User Input
st.sidebar.header("Select a Distribution")
distribution_type = st.sidebar.selectbox(
    "Choose the type of probability distribution:",
    ("Poisson", "Normal", "Binomial")
)

# Common Configurations for Plot
fig, ax = plt.subplots(figsize=(10, 5))

# Distribution-Specific Configurations
if distribution_type == "Poisson":
    st.sidebar.subheader("Poisson Distribution Parameters")
    mean = st.sidebar.slider("Mean (λ):", 1, 20, 4)
    days = np.arange(0, 20)
    poisson_probs = poisson.pmf(days, mean)

    # Plot Poisson
    ax.bar(days, poisson_probs, alpha=0.7, color="blue", label=f"λ = {mean}")
    ax.set_xlabel("Number of Events")
    ax.set_ylabel("Probability")
    ax.set_title("Poisson Distribution")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"### Insights for Poisson Distribution:\n- Mean (λ): {mean}\n- Great for modeling events over time, such as bug reports or support tickets.")

elif distribution_type == "Normal":
    st.sidebar.subheader("Normal Distribution Parameters")
    mean = st.sidebar.slider("Mean (μ):", 0, 1000, 300)
    std_dev = st.sidebar.slider("Standard Deviation (σ):", 1, 300, 50)
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
    normal_probs = norm.pdf(x, mean, std_dev)

    # Plot Normal
    ax.plot(x, normal_probs, label=f"μ = {mean}, σ = {std_dev}", color="green")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Normal Distribution")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"### Insights for Normal Distribution:\n- Mean (μ): {mean}, Standard Deviation (σ): {std_dev}\n- Useful for modeling continuous data like user session times.")

elif distribution_type == "Binomial":
    st.sidebar.subheader("Binomial Distribution Parameters")
    trials = st.sidebar.slider("Number of Trials (n):", 1, 100, 10)
    probability = st.sidebar.slider("Probability of Success (p):", 0.0, 1.0, 0.5)
    outcomes = np.arange(0, trials + 1)
    binom_probs = binom.pmf(outcomes, trials, probability)

    # Plot Binomial
    ax.bar(outcomes, binom_probs, alpha=0.7, color="purple", label=f"n = {trials}, p = {probability}")
    ax.set_xlabel("Number of Successes")
    ax.set_ylabel("Probability")
    ax.set_title("Binomial Distribution")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"### Insights for Binomial Distribution:\n- Trials (n): {trials}, Probability of Success (p): {probability}\n- Ideal for modeling binary outcomes like feature adoption in A/B tests.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by a **Product Manager** exploring **Data Science** insights.")
