"""Streamlit Web界面"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import get_config
from src.data.data_manager import DataManager
from src.strategies.moving_average_strategy import MovingAverageStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.strategy_portfolio import StrategyPortfolio
from src.backtest.backtest_engine import BacktestEngine
from src.risk.risk_manager import RiskManager
from src.optimization.parameter_optimizer import ParameterOptimizer
from src.visualization.plotter import Plotter
import plotly.graph_objects as go
import plotly.express as px

# 页面配置
st.set_page_config(
    page_title="量化交易系统",
    page_icon="📈",
    layout="wide"
)

st.title("📈 量化交易系统")
st.sidebar.title("导航")

# 侧边栏导航
page = st.sidebar.selectbox(
    "选择功能",
    ["回测分析", "参数优化", "实时信号", "策略组合", "关于"]
)

# 数据源配置
st.sidebar.header("数据源配置")
data_source = st.sidebar.selectbox("数据源", ["akshare", "tushare", "yfinance"])
tushare_token = None
if data_source == "tushare":
    tushare_token = st.sidebar.text_input("Tushare Token (可选)", type="password")

# 回测配置
st.sidebar.header("回测配置")
initial_capital = st.sidebar.number_input("初始资金", value=100000, min_value=1000, step=10000)
commission = st.sidebar.number_input("手续费率", value=0.001, min_value=0.0, max_value=0.01, step=0.0001)

# 风险控制配置
st.sidebar.header("风险控制")
enable_risk = st.sidebar.checkbox("启用风险控制", value=False)
stop_loss = st.sidebar.number_input("止损比例", value=0.05, min_value=0.0, max_value=0.5, step=0.01, disabled=not enable_risk)
take_profit = st.sidebar.number_input("止盈比例", value=None, min_value=0.0, max_value=2.0, step=0.05, disabled=not enable_risk)

if page == "回测分析":
    st.header("📊 回测分析")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("股票代码", value="000001")
    with col2:
        start_date = st.date_input("开始日期", value=pd.to_datetime("2023-01-01"))
    with col3:
        end_date = st.date_input("结束日期", value=pd.to_datetime("2024-01-01"))
    
    strategy_type = st.selectbox("策略类型", ["双均线", "RSI", "MACD"])
    
    # 策略参数
    if strategy_type == "双均线":
        col1, col2 = st.columns(2)
        with col1:
            short_window = st.number_input("短期均线", value=5, min_value=1, max_value=50)
        with col2:
            long_window = st.number_input("长期均线", value=20, min_value=1, max_value=200)
        strategy = MovingAverageStrategy(params={'short_window': short_window, 'long_window': long_window})
    
    elif strategy_type == "RSI":
        col1, col2, col3 = st.columns(3)
        with col1:
            rsi_period = st.number_input("RSI周期", value=14, min_value=1, max_value=50)
        with col2:
            oversold = st.number_input("超卖阈值", value=30, min_value=0, max_value=50)
        with col3:
            overbought = st.number_input("超买阈值", value=70, min_value=50, max_value=100)
        strategy = RSIStrategy(params={'rsi_period': rsi_period, 'oversold': oversold, 'overbought': overbought})
    
    elif strategy_type == "MACD":
        col1, col2, col3 = st.columns(3)
        with col1:
            fast = st.number_input("快线周期", value=12, min_value=1, max_value=50)
        with col2:
            slow = st.number_input("慢线周期", value=26, min_value=1, max_value=100)
        with col3:
            signal = st.number_input("信号线周期", value=9, min_value=1, max_value=50)
        strategy = MACDStrategy(params={'fast_period': fast, 'slow_period': slow, 'signal_period': signal})
    
    if st.button("开始回测", type="primary"):
        with st.spinner("正在获取数据..."):
            try:
                # 获取数据
                data_manager = DataManager(source=data_source, token=tushare_token if tushare_token else None)
                data = data_manager.get_data(
                    symbol=str(symbol),
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                
                if data.empty:
                    st.error("数据获取失败，请检查股票代码和数据源配置")
                else:
                    st.success(f"成功获取 {len(data)} 条数据")
                    
                    # 生成信号
                    signals = strategy.generate_signals(data)
                    
                    # 创建风险管理器
                    risk_manager = None
                    if enable_risk:
                        risk_manager = RiskManager(
                            stop_loss=stop_loss,
                            take_profit=take_profit if take_profit else None
                        )
                    
                    # 运行回测
                    engine = BacktestEngine(
                        initial_capital=initial_capital,
                        commission=commission,
                        risk_manager=risk_manager
                    )
                    results = engine.run(data, signals, strategy, symbol=str(symbol))
                    
                    # 计算性能指标
                    metrics = engine.get_performance_metrics(results)
                    
                    # 显示结果
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("总收益率", f"{metrics['total_return']*100:.2f}%")
                    with col2:
                        st.metric("年化收益率", f"{metrics['annual_return']*100:.2f}%")
                    with col3:
                        st.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
                    with col4:
                        st.metric("最大回撤", f"{metrics['max_drawdown']*100:.2f}%")
                    
                    # 可视化
                    fig = go.Figure()
                    
                    # 价格和信号
                    fig.add_trace(go.Scatter(
                        x=results['date'],
                        y=results['close'],
                        mode='lines',
                        name='收盘价',
                        line=dict(color='blue', width=1)
                    ))
                    
                    # 买卖点
                    buy_signals = results[results['signal'] == 1]
                    sell_signals = results[results['signal'] == -1]
                    
                    if not buy_signals.empty:
                        fig.add_trace(go.Scatter(
                            x=buy_signals['date'],
                            y=buy_signals['close'],
                            mode='markers',
                            name='买入',
                            marker=dict(symbol='triangle-up', size=10, color='red')
                        ))
                    
                    if not sell_signals.empty:
                        fig.add_trace(go.Scatter(
                            x=sell_signals['date'],
                            y=sell_signals['close'],
                            mode='markers',
                            name='卖出',
                            marker=dict(symbol='triangle-down', size=10, color='green')
                        ))
                    
                    fig.update_layout(
                        title="价格走势与交易信号",
                        xaxis_title="日期",
                        yaxis_title="价格",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 组合价值
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=results['date'],
                        y=results['portfolio_value'],
                        mode='lines',
                        name='组合价值',
                        line=dict(color='green', width=2)
                    ))
                    fig2.update_layout(
                        title="组合价值变化",
                        xaxis_title="日期",
                        yaxis_title="价值 (元)",
                        height=300
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # 交易记录
                    if engine.trades:
                        st.subheader("交易记录")
                        trades_df = pd.DataFrame(engine.trades)
                        st.dataframe(trades_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"回测失败: {str(e)}")

elif page == "参数优化":
    st.header("🔧 参数优化")
    
    st.info("使用网格搜索优化策略参数")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("股票代码", value="000001", key="opt_symbol")
    with col2:
        start_date = st.date_input("开始日期", value=pd.to_datetime("2023-01-01"), key="opt_start")
    with col3:
        end_date = st.date_input("结束日期", value=pd.to_datetime("2024-01-01"), key="opt_end")
    
    strategy_type = st.selectbox("策略类型", ["双均线", "RSI"], key="opt_strategy")
    
    if strategy_type == "双均线":
        st.subheader("参数范围")
        col1, col2 = st.columns(2)
        with col1:
            short_range = st.text_input("短期均线范围", value="5,10,15,20", help="逗号分隔，如: 5,10,15")
        with col2:
            long_range = st.text_input("长期均线范围", value="20,30,40,50", help="逗号分隔，如: 20,30,40")
        
        if st.button("开始优化", type="primary"):
            with st.spinner("正在优化参数..."):
                try:
                    # 解析参数范围
                    short_windows = [int(x.strip()) for x in short_range.split(',')]
                    long_windows = [int(x.strip()) for x in long_range.split(',')]
                    
                    # 获取数据
                    data_manager = DataManager(source=data_source, token=tushare_token if tushare_token else None)
                    data = data_manager.get_data(
                        symbol=str(symbol),
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d")
                    )
                    
                    if data.empty:
                        st.error("数据获取失败")
                    else:
                        # 参数优化
                        optimizer = ParameterOptimizer(
                            initial_capital=initial_capital,
                            commission=commission
                        )
                        
                        param_grid = {
                            'short_window': short_windows,
                            'long_window': long_windows
                        }
                        
                        result = optimizer.grid_search(
                            MovingAverageStrategy,
                            data,
                            param_grid,
                            metric='sharpe_ratio'
                        )
                        
                        st.success("优化完成！")
                        st.subheader("最优参数")
                        st.json(result['best_params'])
                        st.subheader("最优性能指标")
                        st.json({k: round(v, 4) if isinstance(v, float) else v 
                                for k, v in result['best_metrics'].items()})
                        
                        # 显示前10个结果
                        st.subheader("Top 10 参数组合")
                        top_results = optimizer.get_top_results(10, 'sharpe_ratio')
                        top_df = pd.DataFrame([
                            {
                                '短期均线': r['params']['short_window'],
                                '长期均线': r['params']['long_window'],
                                '夏普比率': round(r['metrics']['sharpe_ratio'], 4),
                                '总收益率': f"{r['metrics']['total_return']*100:.2f}%",
                                '最大回撤': f"{r['metrics']['max_drawdown']*100:.2f}%"
                            }
                            for r in top_results
                        ])
                        st.dataframe(top_df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"优化失败: {str(e)}")

elif page == "实时信号":
    st.header("⚡ 实时信号")
    
    symbol = st.text_input("股票代码", value="000001")
    strategy_type = st.selectbox("策略类型", ["双均线", "RSI", "MACD"], key="realtime_strategy")
    
    # 策略参数（简化版）
    if strategy_type == "双均线":
        strategy = MovingAverageStrategy()
    elif strategy_type == "RSI":
        strategy = RSIStrategy()
    else:
        strategy = MACDStrategy()
    
    if st.button("生成信号", type="primary"):
        with st.spinner("正在生成信号..."):
            try:
                from src.signals.realtime_signal import RealtimeSignalGenerator
                
                data_manager = DataManager(source=data_source, token=tushare_token if tushare_token else None)
                generator = RealtimeSignalGenerator(strategy, data_manager)
                signal = generator.generate_signal(str(symbol))
                
                if 'error' in signal:
                    st.error(signal['error'])
                else:
                    st.success(f"信号生成成功！")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("信号", signal['signal_text'])
                    with col2:
                        st.metric("当前价格", f"{signal['price']:.2f}")
                    with col3:
                        st.metric("日期", signal['date'])
                    
                    # 显示详细信息
                    st.json(signal)
                    
            except Exception as e:
                st.error(f"生成信号失败: {str(e)}")

elif page == "策略组合":
    st.header("🎯 策略组合")
    st.info("组合多个策略，提高稳定性")
    
    st.warning("功能开发中...")

else:
    st.header("📖 关于")
    st.markdown("""
    ## 量化交易系统
    
    一个功能完整的量化交易回测系统，支持：
    
    - ✅ 多数据源支持（akshare, tushare, yfinance）
    - ✅ 多种策略（双均线、RSI、MACD）
    - ✅ 参数优化（网格搜索）
    - ✅ 风险控制（止损、止盈、仓位管理）
    - ✅ 策略组合
    - ✅ 实时信号生成
    - ✅ Web可视化界面
    
    ### 使用说明
    
    1. 在侧边栏配置数据源和回测参数
    2. 选择策略类型和参数
    3. 点击"开始回测"查看结果
    
    ### 注意事项
    
    - 本系统仅用于学习和研究
    - 回测结果不代表未来表现
    - 实盘交易需谨慎
    """)

