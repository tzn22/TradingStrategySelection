"""
Validation Script

Loads trained model, runs on validation data, computes metrics and prints statistics.

Usage:
    python src/scripts/validate.py --model-path trained_models/agent_sac_russel.zip \
                                     --data-path prepared_csv/russel1000_trade_data.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO, SAC, A2C, DDPG, TD3
from preprocessing.df_to_numpy import df_to_numpy
from environment.numpy_stock_env import NumpyStockTradingEnv


def load_model(model_path: str, device: str = 'cpu'):
    """Load trained model from file"""
    print(f"Loading model from: {model_path}")

    # Check the type of model baed on name
    model_name = Path(model_path).stem.lower()

    if 'ppo' in model_name:
        model = PPO.load(model_path, device=device)
        print(f"  Model type: PPO")
    elif 'sac' in model_name:
        model = SAC.load(model_path, device=device)
        print(f"  Model type: SAC")
    elif 'a2c' in model_name:
        model = A2C.load(model_path, device=device)
        print(f"  Model type: A2C")
    elif 'ddpg' in model_name:
        model = DDPG.load(model_path, device=device)
        print(f"  Model type: DDPG")
    elif 'td3' in model_name:
        model = TD3.load(model_path, device=device)
        print(f"  Model type: TD3")
    else:
        # Пробуем SAC по умолчанию
        model = SAC.load(model_path, device=device)
        print(f"  Model type: SAC (default)")

    return model


def run_validation(model, environment, deterministic: bool = True):
    """
    Run model on validation environment and collect results

    Returns:
        account_values: list of portfolio values at each step
        actions: list of actions taken
        dates: list of dates
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    print(f"\nRunning validation...")
    print(f"  Environment steps: {environment.prices.shape[0]}")
    print(f"  Number of stocks: {environment.N}")
    print(f"  Initial capital: ${environment.initial_amount:,.2f}")

    # Wrap environment
    test_env = DummyVecEnv([lambda: environment])
    test_obs = test_env.reset()

    max_steps = environment.prices.shape[0] - 1

    account_values = [environment.initial_amount]
    actions_list = []

    for step in range(max_steps):
        action, _states = model.predict(test_obs, deterministic=deterministic)
        test_obs, rewards, dones, info = test_env.step(action)

        # Collect data
        if len(info) > 0:
            account_values.append(info[0].get('total_assets', account_values[-1]))
            actions_list.append(action[0])

        if dones[0]:
            print(f"  Episode ended at step {step}")
            break

    print(f"  Validation completed: {len(account_values)} steps")

    return account_values, actions_list, environment.dates if hasattr(environment, 'dates') else None


def calculate_metrics(account_values: list, initial_capital: float):
    """
    Calculate performance metrics

    Returns dict with metrics
    """
    values = np.array(account_values)

    # Basic metrics
    final_value = values[-1]
    total_return = (final_value - initial_capital) / initial_capital

    # Returns
    returns = np.diff(values) / values[:-1]
    returns = returns[~np.isnan(returns)]  # Remove NaN

    if len(returns) == 0:
        return {
            'final_value': final_value,
            'total_return_pct': total_return * 100,
            'error': 'No valid returns calculated'
        }

    # Annualized metrics (assuming daily data, 252 trading days)
    mean_return = np.mean(returns)
    annualized_return = (1 + mean_return) ** 252 - 1

    # Volatility
    volatility = np.std(returns)
    annualized_volatility = volatility * np.sqrt(252)

    # Sharpe Ratio (assuming risk-free rate = 0)
    sharpe_ratio = (mean_return / volatility * np.sqrt(252)) if volatility > 0 else 0

    # Max Drawdown
    cummax = np.maximum.accumulate(values)
    drawdown = (cummax - values) / cummax
    max_drawdown = np.max(drawdown)

    # Calmar Ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino_ratio = (mean_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Win rate
    win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0

    return {
        'initial_value': initial_capital,
        'final_value': final_value,
        'total_return_pct': total_return * 100,
        'annualized_return_pct': annualized_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown_pct': max_drawdown * 100,
        'volatility_annual_pct': annualized_volatility * 100,
        'win_rate_pct': win_rate * 100,
        'num_periods': len(returns),
        'avg_daily_return_pct': mean_return * 100,
    }


def print_metrics(metrics: dict, model_name: str):
    """Print metrics in a nice format"""
    print("\n" + "=" * 70)
    print(f"{'VALIDATION RESULTS':^70}")
    print("=" * 70)
    print(f"\nModel: {model_name}")
    print(f"Validation periods: {metrics.get('num_periods', 'N/A')}")

    if 'error' in metrics:
        print(f"\n⚠ Error: {metrics['error']}")
        return

    print("\n" + "-" * 70)
    print(f"{'PORTFOLIO PERFORMANCE':^70}")
    print("-" * 70)
    print(f"  Initial Capital:        ${metrics['initial_value']:>15,.2f}")
    print(f"  Final Value:            ${metrics['final_value']:>15,.2f}")
    print(f"  Total Return:           {metrics['total_return_pct']:>15.2f}%")
    print(f"  Annualized Return:      {metrics['annualized_return_pct']:>15.2f}%")

    print("\n" + "-" * 70)
    print(f"{'RISK METRICS':^70}")
    print("-" * 70)
    print(f"  Max Drawdown:           {metrics['max_drawdown_pct']:>15.2f}%")
    print(f"  Annual Volatility:      {metrics['volatility_annual_pct']:>15.2f}%")
    print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:>15.2f}")
    print(f"  Sortino Ratio:          {metrics['sortino_ratio']:>15.2f}")
    print(f"  Calmar Ratio:           {metrics['calmar_ratio']:>15.2f}")

    print("\n" + "-" * 70)
    print(f"{'TRADING STATS':^70}")
    print("-" * 70)
    print(f"  Win Rate:               {metrics['win_rate_pct']:>15.2f}%")
    print(f"  Avg Daily Return:       {metrics['avg_daily_return_pct']:>15.4f}%")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Validate trained RL trading model')

    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model (e.g., trained_models/agent_sac.zip)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to validation data CSV')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run model on')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic actions')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Path to save results CSV (optional)')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        sys.exit(1)

    print(f"Starting validation...")
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.data_path}")
    print(f"  Device: {args.device}")

    # Load model
    model = load_model(args.model_path, device=args.device)

    # Load data
    print(f"\nLoading validation data...")
    trade = pd.read_csv(args.data_path)

    if trade.columns[0] in ['Unnamed: 0', '']:
        trade = trade.set_index(trade.columns[0])
        trade.index.names = ['']

    print(f"  Data shape: {trade.shape}")
    print(f"  Stocks: {trade['tic'].nunique()}")
    print(f"  Date range: {trade['date'].min()} to {trade['date'].max()}")

    # Prepare environment
    tic_list = sorted(trade['tic'].unique().tolist())

    INDICATORS = [
        'day', 'macd', 'boll_ub', 'boll_lb', 'rsi_30',
        'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix'
    ]

    print(f"\nPreparing trading environment...")
    price_array, tech_array, turbulence_array, dates = df_to_numpy(
        trade, tic_list,
        tech_indicator_list=INDICATORS,
        price_col="close",
        turbulence_col="turbulence"
    )

    env_kwargs = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": 0,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": None,
        "stock_dim": len(tic_list),
        "tech_indicator_list": INDICATORS,
        "action_space": len(tic_list),
        "reward_scaling": 1e-4,
        "allow_negative_cash": False,
        "max_stock_holding": np.full(len(tic_list), 1e9),
        "dates": dates,
    }

    trade_env = NumpyStockTradingEnv(**env_kwargs)

    # Run validation
    account_values, actions, dates = run_validation(
        model, trade_env, deterministic=args.deterministic
    )

    # Calculate metrics
    metrics = calculate_metrics(account_values, trade_env.initial_amount)

    # Print results
    print_metrics(metrics, Path(args.model_path).stem)

    # Save results if requested
    if args.save_results:
        results_df = pd.DataFrame({
            'step': range(len(account_values)),
            'portfolio_value': account_values
        })

        if dates is not None and len(dates) >= len(account_values):
            results_df['date'] = dates[:len(account_values)]

        results_df.to_csv(args.save_results, index=False)
        print(f"\nResults saved to: {args.save_results}")


if __name__ == "__main__":
    main()
