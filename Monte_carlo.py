import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

# --- Monte Carlo Stock Price Simulation Function ---

def monte_carlo_stock_price_simulation(S0, mu, sigma, num_days, dt, num_simulations, plot_placeholder, progress_bar_placeholder, update_frequency):
    """
    Performs a Monte Carlo simulation to forecast stock prices using Geometric Brownian Motion,
    and yields intermediate paths for real-time visualization.

    Args:
        S0 (float): Initial stock price.
        mu (float): Expected annual return (drift coefficient).
        sigma (float): Annual volatility (diffusion coefficient).
        num_days (int): Total number of days to project.
        dt (float): Time step size.
        num_simulations (int): Number of independent simulation paths.
        plot_placeholder (streamlit.delta_generator.DeltaGenerator): Streamlit empty container for plot updates.
        progress_bar_placeholder (streamlit.delta_generator.DeltaGenerator): Streamlit empty container for progress bar.
        update_frequency (int): How often to update the plot (every N paths).

    Returns:
        list: A list of numpy arrays, where each array is a simulated price path.
    """
    all_sim_paths_data = [] # To store all generated paths for the final plot
    current_batch_paths = [] # To store paths for the current real-time plot update

    # Initialize Plotly figure layout once. This will be used to create new figures
    # for each real-time update, ensuring consistent layout and performance.
    base_layout_for_realtime_plot = go.Layout(
        title='Monte Carlo Simulation for Stock Price (Real-time)',
        xaxis_title='Days',
        yaxis_title='Simulated Price',
        height=500,
        showlegend=False,
        # Using a darker template for better contrast with white lines, but still professional
        template="plotly_dark", # Changed to plotly_dark for better contrast with whitish lines
        hovermode="x unified", # Nice for exploring paths
        margin=dict(l=40, r=40, t=60, b=40), # Increased margins slightly
        font=dict(size=12) # Default font size for plot elements
    )
    # Note: Initial y-axis range is set dynamically inside the loop as data accumulates.

    for sim_idx in range(num_simulations):
        prices = [S0]
        # Generate random samples for the whole path at once for performance
        Z_values = np.random.standard_normal(num_days)

        for t in range(num_days):
            daily_return_factor = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_values[t])
            prices.append(prices[-1] * daily_return_factor)

        current_path = np.array(prices)
        all_sim_paths_data.append(current_path) # Always accumulate all for final plot
        current_batch_paths.append(current_path) # Add to current batch for real-time plot

        progress_percent = (sim_idx + 1) / num_simulations
        progress_bar_placeholder.progress(progress_percent, text=f"Simulating path {sim_idx + 1} of {num_simulations}...")

        # Update the plot only when update_frequency paths are generated or it's the last path
        if (sim_idx + 1) % update_frequency == 0 or (sim_idx + 1) == num_simulations:
            with plot_placeholder:
                # Create a NEW Figure for each update, copying the layout
                temp_fig = go.Figure(layout=base_layout_for_realtime_plot)

                for path_in_batch in current_batch_paths:
                    temp_fig.add_trace(go.Scatter(
                        x=np.arange(len(path_in_batch)),
                        y=path_in_batch,
                        mode='lines',
                        line=dict(color='rgba(240, 240, 240, 0.7)', width=1), # Whitish lines for real-time paths
                        showlegend=False,
                        hoverinfo='x+y'
                    ))
                
                # Dynamically adjust y-axis range based on the overall range seen so far
                if all_sim_paths_data: 
                    min_val_overall = np.min([np.min(p) for p in all_sim_paths_data])
                    max_val_overall = np.max([np.max(p) for p in all_sim_paths_data])
                    temp_fig.update_yaxes(range=[min_val_overall * 0.95, max_val_overall * 1.05])

                plot_placeholder.plotly_chart(temp_fig, use_container_width=True)
                current_batch_paths = [] # Clear the batch after plotting to only show new paths next time

                time.sleep(0.001) # Small delay to make updates visible

    return all_sim_paths_data # Return all paths for the final comprehensive plot

# --- Streamlit Application ---

def app():
    st.set_page_config(layout="wide", page_title="Monte Carlo Stock Price Simulator")
    st.title("Real-Time Monte Carlo Stock Price Simulation")

    st.write("""
    This application simulates potential future stock price paths using the Monte Carlo method
    and the Geometric Brownian Motion model. Adjust the parameters below to influence
    the stock's projected movements and watch the paths render in real-time.
    """)

    # --- Input Section (Sidebar for parameters) ---
    st.sidebar.header("Simulation Parameters")

    # Parameters moved back to sidebar
    S0 = st.sidebar.number_input("Initial Stock Price ($S_0$)", value=100.0, min_value=0.01, format="%.2f",
                                 help="The starting price of the stock.")
    mu = st.sidebar.number_input("Annualized Expected Return ($\mu$)", value=0.08, format="%.4f",
                                 help="The average annual return of the stock (e.g., 0.1 for 10%).")
    # Default sigma changed to 0.6
    sigma = st.sidebar.number_input("Annualized Volatility ($\sigma$)", value=0.60, min_value=0.001, format="%.4f",
                                    help="The degree of variation of a trading price series over time (e.g., 0.6 for 60%).")
    
    # Default num_days changed to 50
    num_days = st.sidebar.slider("Number of Days to Project", 30, 730, 50, 1, # Default to 50
                                 help="The total number of days into the future for the simulation.")

    dt_option = st.sidebar.selectbox(
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
    dt = dt_map[dt_option]
    st.sidebar.info(f"Using $\Delta t$ = **{dt:.4f}** based on '{dt_option}' for calculations.")

    # Default num_simulations changed to 1000
    num_simulations = st.sidebar.slider("Number of Simulations (Paths)", 100, 3000, 1000, 100, # Default to 1000
                                         help="The total number of independent price paths to generate.")

    update_frequency = st.sidebar.slider("Update Plot Every (N) Paths", 1, min(num_simulations, 200), 10, 1,
                                         help="Controls how often the 'Simulated Price Paths' plot updates during simulation. Lower values provide more frequent visual updates but can slightly slow down very large simulations.")
    
    st.sidebar.markdown("---")
    
    # Initialize session state for button click to prevent re-running on every input change
    if 'run_clicked' not in st.session_state:
        st.session_state.run_clicked = False

    start_simulation = st.sidebar.button("Run Simulation", type="primary")

    if start_simulation:
        st.session_state.run_clicked = True
    
    if st.session_state.run_clicked:
        if sigma <= 0.001:
            st.error("Please ensure Annualized Volatility (sigma) is greater than 0.001 to run the simulation.")
            st.session_state.run_clicked = False # Reset if invalid input
        else:
            # Placeholders for dynamic content
            st.subheader("Simulated Price Paths (Real-time update)")
            progress_bar_placeholder = st.empty()
            plot_placeholder = st.empty() # Placeholder for the real-time plot

            with st.spinner("Generating simulation paths... Please wait."): # Show spinner for overall calculation time
                all_sim_paths_data = monte_carlo_stock_price_simulation(S0, mu, sigma, num_days, dt, num_simulations, plot_placeholder, progress_bar_placeholder, update_frequency)
            
            progress_bar_placeholder.empty() # Clear progress bar after completion
            st.success("Simulation complete!") # Keep this for clear completion message

            # Convert list of arrays to a 2D NumPy array for easier calculations
            all_sim_paths_array = np.array(all_sim_paths_data).T # Transpose to have days as rows, simulations as columns

            # --- Final Plot (now applied to the initial graph placeholder) ---
            # Re-use the plot_placeholder from the real-time section for the final combined plot
            with plot_placeholder: 
                final_fig_combined = go.Figure(layout=go.Layout(
                    title='Simulated Stock Price Paths with Mean and Confidence Intervals',
                    xaxis_title='Days',
                    yaxis_title='Simulated Price',
                    height=600,
                    showlegend=True, # Show legend for mean and percentiles
                    template="plotly_dark", # Changed to plotly_dark for consistency
                    hovermode="x unified",
                    margin=dict(l=40, r=40, t=60, b=40), # Increased margins slightly
                    font=dict(size=12)
                ))

                # Add all individual paths (faded)
                for sim_idx in range(all_sim_paths_array.shape[1]):
                    final_fig_combined.add_trace(go.Scatter(
                        x=np.arange(all_sim_paths_array.shape[0]),
                        y=all_sim_paths_array[:, sim_idx],
                        mode='lines',
                        line=dict(color='rgba(240, 240, 240, 0.1)', width=1), # Faded whitish for overall view
                        name=f'Path {sim_idx+1}',
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                # Add mean path
                mean_path = np.mean(all_sim_paths_array, axis=1)
                final_fig_combined.add_trace(go.Scatter(
                    x=np.arange(len(mean_path)),
                    y=mean_path,
                    mode='lines',
                    line=dict(color='yellow', width=3, dash='dot'), # Yellow for mean
                    name='Mean Path',
                    showlegend=True
                ))

                # Add confidence intervals
                percentile_5th = np.percentile(all_sim_paths_array, 5, axis=1)
                percentile_95th = np.percentile(all_sim_paths_array, 95, axis=1)

                final_fig_combined.add_trace(go.Scatter(
                    x=np.arange(len(percentile_95th)),
                    y=percentile_95th,
                    mode='lines',
                    line=dict(color='red', width=1, dash='dash'), # Red for 95th percentile
                    name='95th Percentile',
                    showlegend=True
                ))
                final_fig_combined.add_trace(go.Scatter(
                    x=np.arange(len(percentile_5th)),
                    y=percentile_5th,
                    mode='lines',
                    line=dict(color='lime', width=1, dash='dash'), # Lime for 5th percentile
                    name='5th Percentile',
                    showlegend=True
                ))
                
                # Adjust y-axis range to fit all data
                min_val_final = np.min(all_sim_paths_array)
                max_val_final = np.max(all_sim_paths_array)
                final_fig_combined.update_yaxes(range=[min_val_final * 0.95, max_val_final * 1.05])

                st.plotly_chart(final_fig_combined, use_container_width=True)

            # --- Summary Statistics ---
            st.markdown("---")
            st.header("Simulation Results Summary") # Changed to header for more prominence
            
            # Adjusted layout for readability
            col1, col2, col3 = st.columns(3) # Use 3 columns for better arrangement
            final_prices = all_sim_paths_array[-1, :] # Last row contains final prices

            with col1:
                st.metric(label="Mean Final Price", value=f"${np.mean(final_prices):.2f}")
                st.metric(label="Median Final Price", value=f"${np.median(final_prices):.2f}")
            with col2:
                st.metric(label="Standard Deviation", value=f"${np.std(final_prices):.2f}")
                st.metric(label="Min Final Price", value=f"${np.min(final_prices):.2f}")
            with col3:
                st.metric(label="Max Final Price", value=f"${np.max(final_prices):.2f}")
                st.metric(label="5th Percentile (VaR)", value=f"${np.percentile(final_prices, 5):.2f}")
                st.metric(label="95th Percentile", value=f"${np.percentile(final_prices, 95):.2f}")

            st.write("") # Add some space

            # Histogram of Final Prices with counts on top
            st.header("Distribution of Final Prices") # Changed to header
            
            # Use go.Histogram for more control over annotations
            hist_fig = go.Figure()
            counts, bins = np.histogram(final_prices, bins=50)
            
            hist_fig.add_trace(go.Bar(
                x=bins, y=counts, 
                marker_color='skyblue', # Brighter color for histogram bars
                name='Frequency',
                showlegend=False
            ))

            # Add annotations for counts on top of bars
            for i in range(len(counts)):
                if counts[i] > 0: # Only add annotation if count is greater than 0
                    hist_fig.add_annotation(
                        x=(bins[i] + bins[i+1]) / 2, # Center of the bar horizontally
                        y=counts[i],
                        text=str(counts[i]),
                        yshift=15, # Increased yshift to move text further above the bar
                        xanchor='center', # Ensure horizontal centering
                        yanchor='bottom', # Anchor to the bottom of the text
                        showarrow=False,
                        font=dict(color="white", size=14) # Increased font size
                    )

            hist_fig.update_layout(
                title='Distribution of Final Prices',
                xaxis_title='Final Price ($)',
                yaxis_title='Frequency',
                bargap=0.1,
                template="plotly_dark", # Changed to plotly_dark for consistency
                margin=dict(l=40, r=40, t=60, b=40), # Increased margins slightly
                font=dict(size=12)
            )
            st.plotly_chart(hist_fig, use_container_width=True)

            # --- Interactable Tables ---
            st.markdown("---")
            st.header("Explore Simulated Data") # Changed to header

            # Table for Final Prices
            st.markdown("#### Final Prices of All Simulated Paths")
            df_final_prices = pd.DataFrame(final_prices, columns=['Final Price ($)'])
            df_final_prices.index.name = 'Path Index'
            st.dataframe(df_final_prices.style.format({"Final Price ($)": "${:.2f}"}), use_container_width=True, height=250)

            # Downloadable Data
            st.markdown("---")
            st.header("Download Simulation Data") # Changed to header
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
        st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to see the results.")

    # --- About Section (Moved back to sidebar bottom) ---
    st.sidebar.markdown("---")
    st.sidebar.header("About the Model") # Changed to header for sidebar section
    st.sidebar.info(
        """
        This simulation uses **Geometric Brownian Motion (GBM)** to model stock prices.
        The formula is:
        $S_{t+\Delta t} = S_t \\cdot e^{(\\mu - \\frac{1}{2}\\sigma^2)\\Delta t + \\sigma\\sqrt{\\Delta t}Z}$

        Where:
        - $S_t$: Stock price at time t
        - $\\mu$: Annualized expected return (drift)
        - $\\sigma$: Annualized volatility (diffusion)
        - $\\Delta t$: Time step (e.g., 1/252 for daily)
        - $Z$: A random variable from a standard normal distribution (mean=0, std dev=1)

        **Assumptions of GBM:**
        1.  Stock prices follow a random walk.
        2.  Log returns are normally distributed.
        3.  Volatility is constant over time.
        4.  No dividends or transaction costs.
        """
    )
    st.sidebar.caption("Built with Streamlit and Plotly")


if __name__ == "__main__":
    app()
