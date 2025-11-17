"""主程序入口"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.data.data_manager import DataManager
from src.strategies.moving_average_strategy import MovingAverageStrategy
from src.backtest.backtest_engine import BacktestEngine
from src.visualization.plotter import Plotter


def main():
    """主函数"""
    # 加载配置
    config = get_config()
    
    # 设置日志
    log_config = config.get('logging', {})
    logger = setup_logger(
        log_level=log_config.get('level', 'INFO'),
        log_file=log_config.get('file')
    )
    logger.info("=" * 50)
    logger.info("量化交易系统启动")
    logger.info("=" * 50)
    
    # 1. 获取数据
    logger.info("步骤1: 获取数据...")
    data_config = config['data']
    data_manager = DataManager(source=data_config['source'])
    
    data = data_manager.get_data(
        symbol=data_config['symbol'],
        start_date=data_config['start_date'],
        end_date=data_config['end_date'],
        frequency=data_config['frequency']
    )
    
    if data.empty:
        logger.error("数据获取失败，程序退出")
        return
    
    logger.info(f"数据获取成功: {len(data)}条记录")
    
    # 2. 初始化策略
    logger.info("步骤2: 初始化策略...")
    strategy_config = config['strategy']
    strategy = MovingAverageStrategy(params=strategy_config.get('params', {}))
    
    # 3. 生成信号
    logger.info("步骤3: 生成交易信号...")
    signals = strategy.generate_signals(data)
    
    # 4. 运行回测
    logger.info("步骤4: 运行回测...")
    backtest_config = config['backtest']
    engine = BacktestEngine(
        initial_capital=backtest_config['initial_capital'],
        commission=backtest_config['commission'],
        slippage=backtest_config.get('slippage', 0.0)
    )
    
    results = engine.run(data, signals, strategy)
    
    # 5. 计算性能指标
    logger.info("步骤5: 计算性能指标...")
    metrics = engine.get_performance_metrics(results)
    
    # 打印结果
    logger.info("=" * 50)
    logger.info("回测结果:")
    logger.info(f"  总收益率: {metrics['total_return']*100:.2f}%")
    logger.info(f"  年化收益率: {metrics['annual_return']*100:.2f}%")
    logger.info(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"  交易次数: {metrics['trade_count']}")
    logger.info(f"  最终价值: {metrics['final_value']:.2f}元")
    logger.info("=" * 50)
    
    # 6. 可视化
    logger.info("步骤6: 生成可视化图表...")
    plotter = Plotter()
    plotter.plot_backtest_results(results, metrics, save_path='backtest_results.png')
    
    logger.info("程序执行完成！")


if __name__ == "__main__":
    main()

