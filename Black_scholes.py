import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(layout="wide", page_title="Black-Scholes Pricing Model")

st.markdown(
    """
    <style>
        div[data-testid*="stAlert"] {
            background-color: transparent !important;
            border: none !important;
            padding: 0px !important;
            margin: 0px !important;
        }

        .stAlert p {
            color: #BBBBBB !important;
        }

        div[data-testid="stError"] p {
            color: #FF4B4B !important;
            font-weight: bold !important;
        }

        div[data-testid="stSuccess"] p {
            color: #00FF00 !important;
            font-weight: bold !important;
        }

        .stNumberInput > div > div > input,
        .stSlider > div > div > div > div > div {
            background-color: #20242B;
            color: white;
            border-radius: 5px;
            border: 1px solid #30343D;
        }

        /* Style for the Calculate Option Prices button (default state) */
        div.stButton > button {
            background-color: #20242B; /* Darker background, matches input fields */
            color: white;
            border-radius: 5px;
            border: 1px solid #30343D;
            padding: 10px 20px;
            font-size: 16px;
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease; /* Smooth transition */
            width: 100%; /* Ensures button stretches */
        }

        /* Hover effect for the Calculate Option Prices button */
        div.stButton > button:hover {
            background-color: #FF4B4B; /* Reddish accent on hover */
            color: white; /* Text remains white */
            border: 1px solid #FF4B4B; /* Border matches hover color */
        }

        /* Focus effect for the Calculate Option Prices button - revert to default state */
        /* This ensures it doesn't stay red after click, but only on hover */
        div.stButton > button:focus {
            background-color: #20242B; /* Revert to default background on focus */
            color: white; /* Text remains white */
            border: 1px solid #30343D; /* Revert to default border on focus */
            outline: none; /* Remove default focus outline */
        }

        /* Active/Pressed state - ensure it reverts quickly */
        div.stButton > button:active {
            background-color: #20242B; /* Revert quickly on active/press */
            color: white;
            border: 1px solid #30343D;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def norm_cdf(x):
    """Calculates the cumulative distribution function (CDF) for a standard normal distribution."""
    return norm.cdf(x)

def norm_pdf(x):
    """Calculates the probability density function (PDF) for a standard normal distribution."""
    return norm.pdf(x)

def black_scholes(S, K, T, r, sigma, option_type):
    """
    Calculates the theoretical price of a European option using the Black-Scholes model.
    Handles options very close to expiration to prevent division by zero or log(0).
    """
    if T <= 0.0001:
        if option_type == 'call':
            return max(0, S - K)
        elif option_type == 'put':
            return max(0, K - S)
        else:
            return 0.0

    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    return price

def call_net_profit_loss(S_at_expiration, K, premium):
    """Calculates the net profit/loss for a long call option."""
    return np.maximum(0, S_at_expiration - K) - premium

def put_net_profit_loss(S_at_expiration, K, premium):
    """Calculates the net profit/loss for a long put option."""
    return np.maximum(0, K - S_at_expiration) - premium

def calculate_greeks(S, K, T, r, sigma, option_type):
    """
    Calculates the Black-Scholes Greeks (Delta, Gamma, Theta, Vega, Rho).
    Handles cases where time to expiration is very small to avoid issues.
    """
    if T <= 0.0001:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        delta = norm_cdf(d1)
    else:
        delta = norm_cdf(d1) - 1

    gamma = norm_pdf(d1) / (S * sigma * np.sqrt(T))

    if option_type == 'call':
        theta = (- (S * norm_pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm_cdf(d2)) / 365
    else:
        theta = (- (S * norm_pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm_cdf(-d2)) / 365

    vega = S * norm_pdf(d1) * np.sqrt(T) / 100

    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm_cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm_cdf(-d2) / 100

    return delta, gamma, theta, vega, rho


st.header("Black-Scholes Pricing Model")

st.markdown(
    """
    Use this interactive tool to calculate theoretical **call** and **put** option prices
    using the **Black-Scholes model**.
    """
)

st.subheader("Input Parameters")

with st.expander("Price Inputs", expanded=True):
    col_price_input_s, col_price_display_s, col_price_input_k, col_price_display_k = st.columns([0.4, 0.1, 0.4, 0.1])
    
    with col_price_input_s:
        # Changed initial value to 100,000.0
        S_input = st.number_input("Current Stock Price (S)", value=100000.0, min_value=0.0, format="%.2f", help="Current price of the underlying stock.", key="bs_S_input")
    with col_price_display_s:
        # Display S_input with comma formatting, no "Current:"
        st.markdown(f"<p style='font-size: 1.1em; color:white; margin-top: 1.8em;'>₹{S_input:,.2f}</p>", unsafe_allow_html=True)

    with col_price_input_k:
        K_input = st.number_input("Strike Price (K)", value=100000.0, min_value=0.0, format="%.2f", help="The price at which the option holder can buy (call) or sell (put) the underlying asset.", key="bs_K_input")
    with col_price_display_k:
        # Display K_input with comma formatting, no "Current:"
        st.markdown(f"<p style='font-size: 1.1em; color:white; margin-top: 1.8em;'>₹{K_input:,.2f}</p>", unsafe_allow_html=True)


with st.expander("Time and Rate Inputs", expanded=True):
    # Reverting to previous compact slider layout with separate columns for sliders and metrics
    col_time_input, col_time_metric, col_vol_input, col_vol_metric, col_rate_input, col_rate_metric = st.columns([0.3, 0.2, 0.3, 0.2, 0.3, 0.2])
    
    with col_time_input:
        T_months = st.slider("Time to Expiration (Months)", min_value=1, max_value=24, value=12, step=1, help="Time remaining until the option expires, in months (up to 2 years).", key="bs_T_months")
        
    T_input = T_months / 12.0 # Convert months to years for Black-Scholes formula

    with col_time_metric:
        # Display only months if less than a year, otherwise years and months
        if T_months < 12:
            time_display_str = f"{T_months} month{'s' if T_months > 1 else ''}"
        else:
            full_years = int(T_months // 12)
            remaining_months = int(T_months % 12)
            if remaining_months == 0:
                time_display_str = f"{full_years} year{'s' if full_years > 1 else ''}"
            else:
                time_display_str = f"{full_years} year{'s' if full_years > 1 else ''} and {remaining_months} month{'s' if remaining_months > 1 else ''}"
        
        st.metric(label="Calculated Time", value=time_display_str)

    with col_vol_input:
        sigma_input = st.slider("Volatility (σ as decimal)", min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f", help="The standard deviation of the stock's returns, representing its price fluctuation (e.g., 0.20 for 20%).", key="bs_sigma_input")
    with col_vol_metric:
        st.metric(label="Volatility", value=f"{sigma_input*100:.1f}%") # Display volatility as percentage

    with col_rate_input:
        # Increased max_value for risk-free rate to 0.20 (20%)
        r_input = st.slider("Risk-Free Rate (r as decimal)", min_value=0.0, max_value=0.20, value=0.05, step=0.001, format="%.3f", help="The theoretical rate of return of an investment with zero risk, often approximated by government bond yields (e.g., 0.05 for 5%).", key="bs_r_input")
    with col_rate_metric:
        st.metric(label="Risk-Free Rate", value=f"{r_input*100:.2f}%") # Display risk-free rate as percentage


st.markdown("---")

if 'bs_calculated' not in st.session_state:
    st.session_state.bs_calculated = False
    st.session_state.call_price = 0.0
    st.session_state.put_price = 0.0
    st.session_state.K_for_plot = 100.0
    st.session_state.initial_S = 100.0
    st.session_state.T_val = 1.0
    st.session_state.r_val = 0.05
    st.session_state.sigma_val = 0.20
    st.session_state.call_greeks = (0.0, 0.0, 0.0, 0.0, 0.0)
    st.session_state.put_greeks = (0.0, 0.0, 0.0, 0.0, 0.0)

col_button, = st.columns([1])
with col_button:
    if st.button("Calculate Option Prices", key="calculate_bs", use_container_width=True):
        if S_input <= 0 or K_input <= 0 or T_input < 0 or sigma_input < 0:
            st.error("All positive numerical inputs (Stock Price, Strike Price, Time to Expiration, Volatility) must be non-negative.")
            st.session_state.bs_calculated = False
        else:
            st.session_state.call_price = black_scholes(S_input, K_input, T_input, r_input, sigma_input, 'call')
            st.session_state.put_price = black_scholes(S_input, K_input, T_input, r_input, sigma_input, 'put')
            
            st.session_state.call_greeks = calculate_greeks(S_input, K_input, T_input, r_input, sigma_input, 'call')
            st.session_state.put_greeks = calculate_greeks(S_input, K_input, T_input, r_input, sigma_input, 'put')

            st.session_state.K_for_plot = K_input
            st.session_state.initial_S = S_input
            st.session_state.T_val = T_input
            st.session_state.r_val = r_input
            st.session_state.sigma_val = sigma_input
            st.session_state.bs_calculated = True

if st.session_state.bs_calculated:
    st.subheader("Black-Scholes Option Prices")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Call Option Price", value=f"₹{st.session_state.call_price:,.2f}") # Formatted with commas
    with col2:
        st.metric(label="Put Option Price", value=f"₹{st.session_state.put_price:,.2f}") # Formatted with commas

    st.markdown(
        """
        *Note: The Black-Scholes model assumes European options, constant volatility,
        no dividends, no transaction costs, and a constant risk-free rate.*
        """
    )
    st.markdown("---")

    st.subheader("Option Payoff Chart (Net Profit/Loss)")
    
    K_plot = st.session_state.K_for_plot
    call_premium = st.session_state.call_price
    put_premium = st.session_state.put_price
    current_S = st.session_state.initial_S
    T_val = st.session_state.T_val
    r_val = st.session_state.r_val
    sigma_val = st.session_state.sigma_val

    min_S_range = 0
    max_S_range = max(current_S * 2.0, K_plot * 2.0, 200)
    S_range = np.linspace(min_S_range, max_S_range, 400)

    col_toggle_left, col_toggle_right_spacer = st.columns([0.4, 0.6])
    with col_toggle_left:
        st.radio(
            "",
            ('Call Option Payoff', 'Put Option Payoff'),
            key="payoff_toggle",
            horizontal=True
        )

    fig = go.Figure()

    if st.session_state.payoff_toggle == 'Call Option Payoff':
        net_payoffs = call_net_profit_loss(S_range, K_plot, call_premium)
        bep = K_plot + call_premium
        title_text = 'Long Call Option Net Profit/Loss at Expiration'
        option_details_header = "**Call Option Details:**"
        premium_val = call_premium
        
        intrinsic_value = max(0, current_S - K_plot)
        time_value = max(0, premium_val - intrinsic_value)
        
        d1_pop = (np.log(current_S / K_plot) + (r_val + (sigma_val**2) / 2) * T_val) / (sigma_val * np.sqrt(T_val))
        d2_pop = d1_pop - sigma_val * np.sqrt(T_val)
        pop = norm_cdf(d2_pop) * 100

        max_profit_str = "Unlimited"
        max_loss_str = f"₹{premium_val:,.2f} (Capped)" # Formatted with commas
        
        if current_S > K_plot: moneyness = "In-The-Money (ITM)"
        elif current_S < K_plot: moneyness = "Out-of-The-Money (OTM)"
        else: moneyness = "At-The-Money (ATM)"

        delta, gamma, theta, vega, rho = st.session_state.call_greeks

    else:
        net_payoffs = put_net_profit_loss(S_range, K_plot, put_premium)
        bep = K_plot - put_premium
        title_text = 'Long Put Option Net Profit/Loss at Expiration'
        option_details_header = "**Put Option Details:**"
        premium_val = put_premium

        intrinsic_value = max(0, K_plot - current_S)
        time_value = max(0, premium_val - intrinsic_value)
        
        d1_pop = (np.log(current_S / K_plot) + (r_val + (sigma_val**2) / 2) * T_val) / (sigma_val * np.sqrt(T_val))
        d2_pop = d1_pop - sigma_val * np.sqrt(T_val)
        pop = norm_cdf(-d2_pop) * 100

        max_profit_str = f"₹{max(0, K_plot - premium_val):,.2f} (Capped)" # Formatted with commas
        max_loss_str = f"₹{premium_val:,.2f} (Capped)" # Formatted with commas
        
        if current_S < K_plot: moneyness = "In-The-Money (ITM)"
        elif current_S > K_plot: moneyness = "Out-of-The-Money (OTM)"
        else: moneyness = "At-The-Money (ATM)"

        delta, gamma, theta, vega, rho = st.session_state.put_greeks
    
    fig.add_trace(go.Scatter(
        x=S_range, 
        y=net_payoffs, 
        mode='lines', 
        name='Net Profit/Loss', 
        line=dict(color='white', width=2),
        hovertemplate="<br><b>Stock Price:</b> %{x:,.2f}<br><b>Profit/Loss:</b> %{y:,.2f}<extra></extra>" # Formatted with commas
    ))

    profit_mask = net_payoffs > 0
    fig.add_trace(go.Scatter(
        x=np.append(S_range[profit_mask], S_range[profit_mask][::-1]),
        y=np.append(net_payoffs[profit_mask], np.zeros_like(net_payoffs[profit_mask])[::-1]),
        fill='toself',
        fillcolor='rgba(0,180,0,0.2)',
        mode='none',
        showlegend=False,
        hoverinfo='skip'
    ))

    loss_mask = net_payoffs < 0
    fig.add_trace(go.Scatter(
        x=np.append(S_range[loss_mask], S_range[loss_mask][::-1]),
        y=np.append(net_payoffs[loss_mask], np.zeros_like(net_payoffs[loss_mask])[::-1]),
        fill='toself',
        fillcolor='rgba(180,0,0,0.2)',
        mode='none',
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_shape(type="line",
                  x0=K_plot, y0=net_payoffs.min() * 1.1, x1=K_plot, y1=net_payoffs.max() * 1.1,
                  line=dict(color="gray", width=2, dash="dot"),
                  name=f'Strike Price (₹{K_plot:,.2f})') # Formatted with commas
    
    if S_range.min() <= bep <= S_range.max():
        fig.add_shape(type="line",
                        x0=bep, y0=net_payoffs.min() * 1.1, x1=bep, y1=net_payoffs.max() * 1.1,
                        line=dict(color="lightblue", width=2, dash="dash"),
                        name=f'BEP (₹{bep:,.2f})') # Formatted with commas
    
    fig.update_layout(title_text=title_text)
    
    y_min_final = net_payoffs.min() * 1.1
    y_max_final = net_payoffs.max() * 1.1

    if y_min_final > 0: y_min_final = -0.01
    if y_max_final < 0: y_max_final = 0.01

    fig.update_layout(
        template="plotly_dark",
        xaxis_title='Stock Price at Expiration',
        yaxis_title='Profit / Loss',
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)', zerolinecolor='white'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)', zerolinecolor='white', range=[y_min_final, y_max_final]),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)', bordercolor='white', borderwidth=1),
        margin=dict(l=50, r=50, t=50, b=50),
        shapes=[dict(
            type="line",
            x0=S_range.min(), y0=0, x1=S_range.max(), y1=0,
            line=dict(color='white', width=1)
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(option_details_header)
    
    detail_col1, detail_col2, detail_col3 = st.columns(3)

    label_style = "font-size: 0.9em; color:#BBBBBB;"
    value_style = "font-size: 1.0em; color:white;"

    with detail_col1:
        st.markdown(f'<span style="{label_style}">**Premium Paid:**</span> <span style="{value_style}">₹{premium_val:,.2f}</span>', unsafe_allow_html=True) # Formatted with commas
        st.markdown(f'<span style="{label_style}">**Break-Even Point (BEP):**</span> <span style="{value_style}">₹{bep:,.2f}</span>', unsafe_allow_html=True) # Formatted with commas
        st.markdown(f'<span style="{label_style}">**Max Loss:**</span> <span style="{value_style}">{max_loss_str}</span>', unsafe_allow_html=True)
    
    with detail_col2:
        st.markdown(f'<span style="{label_style}">**Max Profit:**</span> <span style="{value_style}">{max_profit_str}</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="{label_style}">**Intrinsic Value:**</span> <span style="{value_style}">₹{intrinsic_value:,.2f}</span>', unsafe_allow_html=True) # Formatted with commas
        st.markdown(f'<span style="{label_style}">**Time Value:**</span> <span style="{value_style}">₹{time_value:,.2f}</span>', unsafe_allow_html=True) # Formatted with commas
    
    with detail_col3:
        st.markdown(f'<span style="{label_style}">**Moneyness:**</span> <span style="{value_style}">{moneyness}</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="{label_style}">**Probability of Profit (PoP):**</span> <span style="{value_style}">{pop:.2f}%</span>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<h3><span style='color:white;'>Option Greeks</span></h3>", unsafe_allow_html=True)
    
    if st.session_state.payoff_toggle == 'Call Option Payoff':
        delta, gamma, theta, vega, rho = st.session_state.call_greeks
    else:
        delta, gamma, theta, vega, rho = st.session_state.put_greeks

    greeks_col1, greeks_col2, greeks_col3 = st.columns(3)

    with greeks_col1:
        st.markdown(f'<span style="{label_style}">**Delta:**</span> <span style="{value_style}">{delta:.2f}</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="{label_style}">**Gamma:**</span> <span style="{value_style}">{gamma:.2f}</span>', unsafe_allow_html=True)
    
    with greeks_col2:
        st.markdown(f'<span style="{label_style}">**Theta (Daily):**</span> <span style="{value_style}">{theta:.2f}</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="{label_style}">**Vega:**</span> <span style="{value_style}">{vega:.2f}</span>', unsafe_allow_html=True)
    
    with greeks_col3:
        st.markdown(f'<span style="{label_style}">**Rho:**</span> <span style="{value_style}">{rho:.2f}</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="{label_style}">*Note: Vega & Rho per 1 unit change.*</span>', unsafe_allow_html=True)

else:
    st.info("Enter parameters in the input sections above and click 'Calculate Option Prices' to see results, including option prices and payoff charts.")

st.markdown("---")
st.header("About the Model")
st.info(
    """
    The **Black-Scholes model** is a mathematical model for pricing an options contract.

    **Formula for Call Option ($C$):**
    $C = S N(d_1) - K e^{-rT} N(d_2)$

    **Formula for Put Option ($P$):**
    $P = K e^{-rT} N(-d_2) - S N(-d_1)$


    **Key Assumptions of Black-Scholes:**
    1.  **European-style options:** Can only be exercised at expiration.
    2.  **No dividends:** The underlying stock pays no dividends during the option's life.
    3.  **Efficient markets:** Stock price movements cannot be predicted.
    4.  **No transaction costs:** Buying or selling options incurs no fees.
    5.  **Constant risk-free rate:** The risk-free rate remains constant.
    6.  **Constant volatility:** The volatility of the underlying asset is constant.
    7.  **Lognormal distribution:** Stock prices follow a lognormal distribution.
    """
)
st.caption("Built with Streamlit and Plotly")