import yfinance as yf
import pandas as pd
import numpy as np

# Fonction pour récupérer les données historiques
def get_historical_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        print("No data found. Please check your ticker and date range.")
        return None
    return data['Adj Close']

# Calcul des rendements journaliers
def calculate_daily_returns(prices):
    returns = prices.pct_change().dropna()
    return returns

# VaR Monte Carlo
def var_monte_carlo(returns, confidence_level=0.95, num_simulations=10000):
    mean = np.mean(returns)
    std_dev = np.std(returns)
    
    simulated_returns = np.random.normal(mean, std_dev, num_simulations)
    var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
    
    return var

# Expected Shortfall
def expected_shortfall(returns, var):
    es = -returns[returns < -var].mean()
    return es

# Stress Testing: Apply selected financial crises with tailored shocks
def apply_stress_test(prices, crises_shocks, asset_type, periods=1):
    stressed_prices = prices.copy()
    
    # Calculate the shock per period
    for crisis, shock in crises_shocks[asset_type].items():
        print(f"Applying {crisis} shock ({shock*100:.2f}% total) for asset type {asset_type}.")
        shock_per_period = shock / periods
        for period in range(periods):
            stressed_prices *= (1 + shock_per_period)  # Applying shock incrementally
            
    return stressed_prices

# Backtesting function to compare actual vs predicted VaR
def backtest_var(prices, var, confidence_level=0.95):
    daily_returns = calculate_daily_returns(prices)
    var_violations = daily_returns[daily_returns < -var]
    
    total_days = len(daily_returns)
    var_days = len(var_violations)
    expected_var_days = int((1 - confidence_level) * total_days)
    
    print(f"Backtesting Results:\n")
    print(f"Total Days: {total_days}")
    print(f"VaR Days (where loss exceeded VaR): {var_days}")
    print(f"Expected VaR Days (at confidence level {confidence_level*100:.2f}%): {expected_var_days}")
    
    if var_days > expected_var_days:
        print("Warning: More losses than expected have exceeded the VaR.")
    else:
        print("VaR model performance is within expected bounds.")

# Interface utilisateur pour les entrées
def terminal_interface():
    print("Welcome to the VaR Calculator V2.")
    
    # Entrée de l'utilisateur pour les données
    ticker = input("Enter the ticker symbol (e.g., AAPL, MSFT, CL=F for crude oil): ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    
    # Récupération des données
    prices = get_historical_data(ticker, start_date, end_date)
    if prices is None:
        return
    
    # Calcul des rendements journaliers
    daily_returns = calculate_daily_returns(prices)
    
    # Monte Carlo VaR et Expected Shortfall
    confidence_level = float(input("Enter confidence level (e.g., 0.95 for 95%): "))
    num_simulations = int(input("Enter the number of simulations for Monte Carlo: "))
    
    var_mc = var_monte_carlo(daily_returns, confidence_level, num_simulations)
    es_mc = expected_shortfall(daily_returns, var_mc)
    
    print(f"\nMonte Carlo VaR (confidence level {confidence_level*100:.2f}%): {var_mc:.4f}")
    print(f"Expected Shortfall (ES): {es_mc:.4f}\n")
    
    # Application des stress tests
    apply_stress_tests = input("Do you want to apply stress tests (top 5 financial crises)? (yes/no): ").lower()
    
    if apply_stress_tests == 'yes':
        # Exemples de chocs de crises financières par type d'actif
        crises_shocks = {
            "stock": {
                "2008 Financial Crisis": -0.50,
                "COVID-19 Market Crash": -0.35,
                "Dot-com Bubble": -0.40,
                "Black Monday 1987": -0.22,
                "Asian Financial Crisis 1997": -0.35
            },
            "commodity": {
                "2008 Financial Crisis": -0.40,
                "COVID-19 Market Crash": -0.25,
                "Dot-com Bubble": -0.30,
                "Black Monday 1987": -0.15,
                "Asian Financial Crisis 1997": -0.30
            },
            "oil": {
                "2008 Financial Crisis": -0.60,
                "COVID-19 Market Crash": -0.45,
                "Dot-com Bubble": -0.35,
                "Black Monday 1987": -0.20,
                "Asian Financial Crisis 1997": -0.40
            }
        }
        
        # Sélection de type d'actif
        asset_type = input("Enter the type of asset (stock, commodity, oil): ").lower()
        
        print("\nAvailable crises:")
        for idx, crisis in enumerate(crises_shocks[asset_type].keys()):
            print(f"{idx+1}. {crisis}")
        
        selected_crises = input("\nEnter the number of crises you want to apply (e.g., 1 3 for 2008 and Dot-com): ").split()
        selected_shocks = {list(crises_shocks[asset_type].keys())[int(i)-1]: list(crises_shocks[asset_type].values())[int(i)-1] for i in selected_crises}
        
        periods = int(input("Enter the number of periods over which to apply the stress (e.g., 3 for 3 days): "))
        stressed_prices = apply_stress_test(prices, {asset_type: selected_shocks}, asset_type, periods)
        stressed_returns = calculate_daily_returns(stressed_prices)
        
        var_stressed = var_monte_carlo(stressed_returns, confidence_level, num_simulations)
        es_stressed = expected_shortfall(stressed_returns, var_stressed)
        
        print(f"\nStressed VaR: {var_stressed:.4f}")
        print(f"Stressed Expected Shortfall (ES): {es_stressed:.4f}\n")
    
    # Option de backtesting
    perform_backtest = input("Do you want to run backtesting on the VaR model? (yes/no): ").lower()
    
    if perform_backtest == 'yes':
        backtest_var(prices, var_mc, confidence_level)

# Lancement de l'interface terminal
if __name__ == "__main__":
    terminal_interface()
