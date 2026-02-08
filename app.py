import streamlit as st
import pandas as pd
import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Future Wealth Planner",
    page_icon="💰",
    layout="wide"
)

# --- HEADER ---
st.title("Future Wealth Planner 💰")
st.write("A tool to forecast your financial future based on detailed inputs.")
st.markdown("---")

# --- SIDEBAR FOR GLOBAL ASSUMPTIONS ---
with st.sidebar:
    st.header("Global Assumptions")
    
    # These inputs affect the entire calculation
    current_age = st.number_input("Your Current Age", min_value=18, max_value=100, value=30, step=1)
    retirement_age = st.number_input("Target Retirement Age", min_value=40, max_value=100, value=65, step=1)
    inflation_rate = st.slider("Assumed Annual Inflation Rate (%)", 0.0, 5.0, 2.5, 0.1)
    investment_return = st.slider("Assumed Annual Investment Return (%)", 0.0, 15.0, 7.0, 0.1)
    
    st.markdown("---")
    
    # The "Calculate" button
    # Using a form helps prevent the app from re-running on every single input change
    with st.form("calculation_form"):
        submitted = st.form_submit_button("Calculate Financial Future")

# --- MAIN PANEL WITH TABS FOR INPUTS ---
# Create the tabs
tab1, tab2, tab3, tab4 = st.tabs(["💰 Income", "📈 Assets & Investments", "💸 Expenses", "🏡 Life Events"])

with tab1:
    st.header("Income Sources")
    st.write("Detail all sources of income, current and future.")
    
    # Primary Job
    st.subheader("Primary Job")
    job_income = st.number_input("Current Annual Salary ($)", min_value=0, value=75000, step=1000)
    job_growth = st.slider("Annual Salary Growth Rate (%)", 0.0, 10.0, 3.0, 0.1)
    
    # Expander for other income sources
    with st.expander("Add Additional Income (Pensions, Social Security, etc.)"):
        st.write("Add any other income streams you anticipate.")
        ss_start_age = st.number_input("Social Security Start Age", min_value=62, max_value=70, value=67)
        ss_monthly_amount = st.number_input("Estimated Monthly Social Security ($)", min_value=0, value=2000, step=100)
        # TODO: Add inputs for pensions, rental income, etc.

with tab2:
    st.header("Assets & Investments")
    st.write("Detail your current investment accounts.")

    st.subheader("Retirement Accounts")
    pretax_401k = st.number_input("Current 401(k) / Traditional IRA Balance ($)", min_value=0, value=50000)
    roth_ira = st.number_input("Current Roth IRA / Roth 401(k) Balance ($)", min_value=0, value=25000)
    
    with st.expander("Add Monthly Contributions"):
        pretax_contrib = st.number_input("Monthly Pre-Tax Contribution ($)", min_value=0, value=500)
        roth_contrib = st.number_input("Monthly Roth Contribution ($)", min_value=0, value=300)

    st.subheader("Other Investments")
    brokerage_account = st.number_input("Taxable Brokerage Account Balance ($)", min_value=0, value=10000)
    # TODO: Add inputs for HSA, real estate, etc.

with tab3:
    st.header("Expenses")
    st.write("Estimate your monthly and future expenses.")
    
    monthly_expenses = st.number_input("Current Monthly Expenses ($)", min_value=0, value=4000, step=100)
    retirement_expenses = st.number_input("Estimated Monthly Expenses in Retirement ($)", min_value=0, value=3000, step=100)

with tab4:
    st.header("Major Life Events")
    st.write("Add one-time financial events, positive or negative.")
    
    # This is a good place to have a dynamic list, but we'll start with one example
    st.subheader("Example: Buying a House")
    house_purchase_age = st.number_input("Age of Purchase", min_value=current_age, max_value=100, value=35)
    house_cost = st.number_input("Cost of Event ($)", min_value=0, value=-50000, step=1000, help="Use a negative number for a cost, positive for a windfall.")


# --- CALCULATION AND REPORTING ---
# This part of the code only runs when the "Calculate" button is pressed
if submitted:
    st.markdown("---")
    st.header("Your Financial Forecast")
    
    st.write(f"Starting at age **{current_age}** and planning for retirement at age **{retirement_age}**...")
    
    # TODO: Build the financial forecasting model here
    # The model will take all the inputs from above and create a year-by-year dataframe.
    
    st.info("The forecasting model is the next step! The chart and report will be generated from its results.")
    
    # Placeholder for the future chart and report
    st.subheader("Projected Portfolio Value")
    # st.line_chart(...)
    
    st.subheader("Retirement Analysis")
    # st.write(...)
