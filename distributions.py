import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, binom

# Streamlit App Title
st.title("Interactive Probability Distributions Explorer")
st.markdown("Explore **Discrete** and **Continuous** Probability Distributions with customizable parameters and actionable insights.")

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
    st.sidebar.markdown("- **Mean (λ)**: The average number of events (e.g., tickets, issues) expected per time period.")
    st.sidebar.markdown("- **Threshold**: The maximum number of events your team can handle before being overwhelmed.")
    mean = st.sidebar.slider("Mean (λ):", 1, 20, 4)
    threshold = st.sidebar.slider("Threshold for Events:", 1, 20, 10)
    days = np.arange(0, 20)
    poisson_probs = poisson.pmf(days, mean)

    # Plot Poisson
    ax.bar(days, poisson_probs, alpha=0.7, color="blue", label=f"λ = {mean}")
    ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
    ax.set_xlabel("Number of Events")
    ax.set_ylabel("Probability")
    ax.set_title("Poisson Distribution")
    ax.legend()
    st.pyplot(fig)

    # Insights Panel
    probability_exceeding = poisson.sf(threshold, mean) * 100
    st.markdown(f"### Insights for Poisson Distribution:\n- Mean (λ): {mean}\n- There is a **{probability_exceeding:.2f}% chance** of exceeding {threshold} events in a day.\n- **Actionable Insight**: Consider allocating resources if exceeding this threshold impacts operations.")

elif distribution_type == "Normal":
    st.sidebar.subheader("Normal Distribution Parameters")
    st.sidebar.markdown("- **Mean (μ)**: The average value (e.g., session length) around which data is centered.")
    st.sidebar.markdown("- **Standard Deviation (σ)**: The spread of the data around the mean; higher values indicate greater variability.")
    st.sidebar.markdown("- **Lower/Upper Bounds**: The range within which most data is expected to fall.")
    mean = st.sidebar.slider("Mean (μ):", 0, 1000, 300)
    std_dev = st.sidebar.slider("Standard Deviation (σ):", 1, 300, 50)
    lower_bound = st.sidebar.slider("Lower Bound for Analysis:", 0, 1000, 250)
    upper_bound = st.sidebar.slider("Upper Bound for Analysis:", 0, 1000, 350)
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
    normal_probs = norm.pdf(x, mean, std_dev)

    # Plot Normal
    ax.plot(x, normal_probs, label=f"μ = {mean}, σ = {std_dev}", color="green")
    ax.axvspan(lower_bound, upper_bound, color="yellow", alpha=0.3, label="Analysis Range")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Normal Distribution")
    ax.legend()
    st.pyplot(fig)

    # Insights Panel
    st.markdown(f"### Insights for Normal Distribution:\n- Mean (μ): {mean}, Standard Deviation (σ): {std_dev}\n- **68% of data** falls within one standard deviation: [{mean - std_dev}, {mean + std_dev}].\n- Analysis Range: {lower_bound} to {upper_bound}.\n- **Actionable Insight**: If many users fall outside this range, consider segmentation for optimization.")

elif distribution_type == "Binomial":
    st.sidebar.subheader("Binomial Distribution Parameters")
    st.sidebar.markdown("- **Number of Trials (n)**: The total number of attempts (e.g., users in an A/B test).")
    st.sidebar.markdown("- **Probability of Success (p)**: The likelihood of a single trial succeeding (e.g., a user adopting a feature).")
    st.sidebar.markdown("- **Success Threshold**: The number of successes required to meet your objective.")
    trials = st.sidebar.slider("Number of Trials (n):", 1, 100, 10)
    probability = st.sidebar.slider("Probability of Success (p):", 0.0, 1.0, 0.5)
    success_threshold = st.sidebar.slider("Success Threshold:", 0, 100, 7)
    outcomes = np.arange(0, trials + 1)
    binom_probs = binom.pmf(outcomes, trials, probability)

    # Plot Binomial
    ax.bar(outcomes, binom_probs, alpha=0.7, color="purple", label=f"n = {trials}, p = {probability}")
    ax.axvline(success_threshold, color="red", linestyle="--", label=f"Threshold = {success_threshold}")
    ax.set_xlabel("Number of Successes")
    ax.set_ylabel("Probability")
    ax.set_title("Binomial Distribution")
    ax.legend()
    st.pyplot(fig)

    # Insights Panel
    cumulative_prob = binom.cdf(success_threshold, trials, probability) * 100
    st.markdown(f"### Insights for Binomial Distribution:\n- Trials (n): {trials}, Probability of Success (p): {probability}\n- **{cumulative_prob:.2f}% chance** of achieving up to {success_threshold} successes.\n- **Actionable Insight**: If the success rate falls below expectations, re-evaluate the experiment parameters or sample size.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by a **Product Manager** exploring **Data Science** insights.")
