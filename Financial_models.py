import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(layout="wide")

st.markdown("""
    <style>
        /* Center tabs and increase spacing */
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
            gap: 250px;
        }

        /* Custom style for Run Simulation button */
        div.stButton > button {
            background-color: #333333; /* Darker background for the button */
            color: white; /* White text for the button */
            border: 1px solid #444444; /* Subtle border */
            width: 100%; /* Make button stretch full width */
        }
        div.stButton > button:hover {
            background-color: #FF4B4B; /* Red on hover */
            color: white;
            border: 1px solid #FF4B4B; /* Red border on hover */
        }

        /* --- Custom Styles for Alerts (st.info, st.error, st.success, st.warning) --- */

        /* Remove background, border, padding, and margin from all Streamlit alert types */
        /* This makes the "boxes" entirely disappear, leaving only the text */
        div[data-testid*="stAlert"] {
            background-color: transparent !important; /* Makes the box background transparent */
            border: none !important; /* Removes the border around the box */
            padding: 0px !important; /* Removes internal padding */
            margin: 0px !important; /* Removes external margin, collapsing space */
        }

        /* Ensure default text color for all alerts is a readable light grey (#BBBBBB) */
        .stAlert p {
            color: #BBBBBB !important; /* Reverted to light grey for general alert text */
        }

        /* Specific styling for st.error messages (e.g., when sigma is too low) */
        /* Text will be red and bold, but without the red background box */
        div[data-testid="stError"] p {
            color: #FF4B4B !important; /* Red text for error messages */
            font-weight: bold !important;
        }

        /* Specific styling for st.success messages */
        div[data-testid="stSuccess"] p {
            color: #00FF00 !important; /* Green text for success messages */
            font-weight: bold !important;
        }

    </style>
""", unsafe_allow_html=True)

def monte_carlo_stock_price_simulation(S0, mu, sigma, num_days, dt, num_simulations, plot_placeholder, progress_bar_placeholder, update_frequency):
    all_sim_paths_data = []
    current_batch_paths = []

    base_layout_for_realtime_plot = go.Layout(
        title='Monte Carlo Simulation for Stock Price (Real-time)',
        xaxis_title='Days',
        yaxis_title='Simulated Price',
        height=500,
        showlegend=False,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(size=12)
    )

    for sim_idx in range(num_simulations):
        prices = [S0]
        Z_values = np.random.standard_normal(num_days)

        for t in range(num_days):
            daily_return_factor = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_values[t])
            prices.append(prices[-1] * daily_return_factor)

        current_path = np.array(prices)
        all_sim_paths_data.append(current_path)
        current_batch_paths.append(current_path)

        progress_percent = (sim_idx + 1) / num_simulations
        progress_bar_placeholder.progress(progress_percent, text=f"Simulating path {sim_idx + 1} of {num_simulations}...")

        if (sim_idx + 1) % update_frequency == 0 or (sim_idx + 1) == num_simulations:
            with plot_placeholder:
                temp_fig = go.Figure(layout=base_layout_for_realtime_plot)

                for path_in_batch in current_batch_paths:
                    temp_fig.add_trace(go.Scatter(
                        x=np.arange(len(path_in_batch)),
                        y=path_in_batch,
                        mode='lines',
                        line=dict(color='rgba(255, 255, 0, 0.7)', width=1),
                        showlegend=False,
                        hoverinfo='x+y'
                    ))

                if all_sim_paths_data:
                    min_val_overall = np.min([np.min(p) for p in all_sim_paths_data])
                    max_val_overall = np.max([np.max(p) for p in all_sim_paths_data])
                    temp_fig.update_yaxes(range=[min_val_overall * 0.95, max_val_overall * 1.05])

                plot_placeholder.plotly_chart(temp_fig, use_container_width=True)
                current_batch_paths = []

                time.sleep(0.001)

    return all_sim_paths_data


st.header("Real-Time Monte Carlo Stock Price Simulation")

st.write("""
This application simulates potential future stock price paths using the Monte Carlo method
and the Geometric Brownian Motion model. Adjust the parameters below to influence
the stock's projected movements and watch the paths render in real-time.
""")

st.subheader("Simulation Parameters")

col_left, col_right = st.columns([0.5, 0.5])

with col_left:
    S0 = st.number_input("Initial Stock Price ($S_0$)", value=100.00, min_value=0.01, format="%.2f",
                            help="The starting price of the stock.")
    mu = st.number_input("Annualized Expected Return ($\mu$)", value=0.08, format="%.4f",
                            help="The average annual return of the stock (e.g., 0.1 for 10%).")
    sigma = st.number_input("Annualized Volatility ($\sigma$)", value=0.60, min_value=0.001, format="%.4f",
                                help="The degree of variation of a trading price series over time (e.6 for 60%).")
    dt_option = st.selectbox(
        "Time Step Frequency ($\Delta t$)",
        options=['Daily (1/252)', 'Weekly (1/52)', 'Monthly (1/12)'],
        index=0,
        help="The frequency of price changes in the simulation. 252 trading days in a year is standard."
    )
    dt_map = {
        'Daily (1/252)': 1/252,
        'Weekly (1/52)': 1/52,
        'Monthly (1/12)': 1/12
    }
    st.info(f"Using $\Delta t$ = **{dt_map[dt_option]:.4f}** based on '{dt_option}' for calculations.")
    dt = dt_map[dt_option]

with col_right:
    num_days = st.slider("Number of Days to Project", 30, 730, 50, 1,
                            help="The total number of days into the future for the simulation.")
    num_simulations = st.slider("Number of Simulations (Paths)", 100, 3000, 1000, 100,
                                    help="The total number of independent price paths to generate.")
    update_frequency = st.slider("Update Plot Every (N) Paths", 1, min(num_simulations, 200), 10, 1,
                                    help="Controls how often the 'Simulated Price Paths' plot updates during simulation. Lower values provide more frequent visual updates but can slightly slow down very large simulations.")


st.markdown("---")

if 'run_clicked' not in st.session_state:
    st.session_state.run_clicked = False

start_simulation = st.button("Run Simulation")

if start_simulation:
    st.session_state.run_clicked = True

if st.session_state.run_clicked:
    if sigma <= 0.001:
        st.error("Please ensure Annualized Volatility (sigma) is greater than 0.001 to run the simulation.")
        st.session_state.run_clicked = False
    else:
        st.subheader("Simulated Price Paths (Real-time update)")
        progress_bar_placeholder = st.empty()
        plot_placeholder = st.empty()

        with st.spinner("Generating simulation paths... Please wait."):
            all_sim_paths_data = monte_carlo_stock_price_simulation(S0, mu, sigma, num_days, dt, num_simulations, plot_placeholder, progress_bar_placeholder, update_frequency)

        progress_bar_placeholder.empty()
        st.success("Simulation complete!")

        with plot_placeholder:
            all_sim_paths_array = np.array(all_sim_paths_data).T
            final_fig_combined = go.Figure(layout=go.Layout(
                title='Simulated Stock Price Paths with Mean and Confidence Intervals',
                xaxis_title='Days',
                yaxis_title='Simulated Price',
                height=600,
                showlegend=True,
                template="plotly_dark",
                hovermode="x unified",
                margin=dict(l=40, r=40, t=60, b=40),
                font=dict(size=12)
            ))

            mean_final_price = np.mean(all_sim_paths_array[-1, :])

            for sim_idx in range(all_sim_paths_array.shape[1]):
                path_color = 'rgba(0, 255, 0, 0.1)' if all_sim_paths_array[-1, sim_idx] > mean_final_price else 'rgba(255, 0, 0, 0.1)'
                final_fig_combined.add_trace(go.Scatter(
                    x=np.arange(all_sim_paths_array.shape[0]),
                    y=all_sim_paths_array[:, sim_idx],
                    mode='lines',
                    line=dict(color=path_color, width=1),
                    name=f'Path {sim_idx+1}',
                    showlegend=False,
                    hoverinfo='skip'
                ))

            mean_path = np.mean(all_sim_paths_array, axis=1)
            final_fig_combined.add_trace(go.Scatter(
                x=np.arange(len(mean_path)),
                y=mean_path,
                mode='lines',
                line=dict(color='yellow', width=3, dash='dot'),
                name='Mean Path',
                showlegend=True
            ))

            percentile_5th = np.percentile(all_sim_paths_array, 5, axis=1)
            percentile_95th = np.percentile(all_sim_paths_array, 95, axis=1)

            final_fig_combined.add_trace(go.Scatter(
                x=np.arange(len(percentile_95th)),
                y=percentile_95th,
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                name='95th Percentile',
                showlegend=True
            ))
            final_fig_combined.add_trace(go.Scatter(
                x=np.arange(len(percentile_5th)),
                y=percentile_5th,
                mode='lines',
                line=dict(color='lime', width=1, dash='dash'),
                name='5th Percentile',
                showlegend=True
            ))

            min_val_final = np.min(all_sim_paths_array)
            max_val_final = np.max(all_sim_paths_array)
            final_fig_combined.update_yaxes(range=[min_val_final * 0.95, max_val_final * 1.05])

            st.plotly_chart(final_fig_combined, use_container_width=True)

        st.markdown("---")
        st.header("Simulation Results Summary")

        col1, col2, col3 = st.columns(3)
        final_prices = all_sim_paths_array[-1, :]

        with col1:
            st.metric(label="Mean Final Price", value=f"₹{np.mean(final_prices):.2f}")
            st.metric(label="Median Final Price", value=f"₹{np.median(final_prices):.2f}")
        with col2:
            st.metric(label="Standard Deviation", value=f"₹{np.std(final_prices):.2f}")
            st.metric(label="Min Final Price", value=f"₹{np.min(final_prices):.2f}")
        with col3:
            st.metric(label="Max Final Price", value=f"₹{np.max(final_prices):.2f}")
            st.metric(label="5th Percentile (VaR)", value=f"₹{np.percentile(final_prices, 5):.2f}")
            st.metric(label="95th Percentile", value=f"₹{np.percentile(final_prices, 95):.2f}")

        st.write("")

        st.header("Distribution of Final Prices")

        hist_fig = go.Figure()
        counts, bins = np.histogram(final_prices, bins=50)

        hist_fig.add_trace(go.Bar(
            x=bins, y=counts,
            marker_color='skyblue',
            name='Frequency',
            showlegend=False
        ))

        for i in range(len(counts)):
            if counts[i] > 0:
                hist_fig.add_annotation(
                    x=(bins[i] + bins[i+1]) / 2,
                    y=counts[i],
                    text=str(counts[i]),
                    yshift=15,
                    xanchor='center',
                    yanchor='bottom',
                    showarrow=False,
                    font=dict(color="white", size=14)
                )

        hist_fig.update_layout(
            title='Distribution of Final Prices',
            xaxis_title='Final Price (₹)',
            yaxis_title='Frequency',
            bargap=0.1,
            template="plotly_dark",
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(size=12)
        )
        st.plotly_chart(hist_fig, use_container_width=True)

        st.markdown("---")
        st.header("Explore Simulated Data")

        st.markdown("#### Final Prices of All Simulated Paths")
        df_final_prices = pd.DataFrame(final_prices, columns=['Final Price (₹)'])
        df_final_prices.index.name = 'Path Index'
        st.dataframe(df_final_prices.style.format({"Final Price (₹)": "₹{:.2f}"}), use_container_width=True, height=250)

        st.markdown("---")
        st.header("Download Simulation Data")
        df_sim_paths = pd.DataFrame(all_sim_paths_array, columns=[f'Sim_{i+1}' for i in range(num_simulations)])
        df_sim_paths.index.name = 'Day'
        csv_data = df_sim_paths.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download All Simulated Paths as CSV",
            data=csv_data,
            file_name="monte_carlo_stock_paths.csv",
            mime="text/csv",
            help="Download a CSV file containing all simulated stock price paths."
        )
else:
    st.info("Adjust the parameters above and click 'Run Simulation' to see the results.")

st.markdown("---")
st.header("About the Model")
st.info(
    """
    This simulation uses **Geometric Brownian Motion (GBM)** to model stock prices.
    The formula is:
    $S_{t+\Delta t} = S_t \\cdot e^{(\\mu - \\frac{1}{2}\\sigma^2)\\Delta t + \\sigma\\sqrt{\\Delta t}Z}$

    **Assumptions of GBM:**
    1. Stock prices follow a random walk.
    2. Log returns are normally distributed.
    3. Volatility is constant over time.
    4. No dividends or transaction costs.
    """
)
st.caption("Built with Streamlit and Plotly")
