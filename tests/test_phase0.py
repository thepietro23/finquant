"""Phase 0 sanity checks — config, seed, logger, metrics all work."""
import os
import sys
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestConfig:
    def test_config_loads(self):
        from src.utils.config import load_config
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert 'seed' in cfg
        assert cfg['seed'] == 42

    def test_config_sections(self):
        from src.utils.config import get_config
        data_cfg = get_config('data')
        assert data_cfg['stocks'] == 'nifty50'
        assert data_cfg['risk_free_rate'] == 0.07

        rl_cfg = get_config('rl')
        assert rl_cfg['algorithm'] == 'PPO'
        assert rl_cfg['max_position'] == 0.20

    def test_config_all_sections_exist(self):
        from src.utils.config import get_config
        cfg = get_config()
        required_sections = ['data', 'features', 'sentiment', 'gnn', 'rl',
                             'gan', 'stress', 'nas', 'fl', 'quantum', 'api']
        for section in required_sections:
            assert section in cfg, f'Missing config section: {section}'


class TestSeed:
    def test_seed_reproducibility(self):
        from src.utils.seed import set_seed
        import torch

        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b), 'Same seed should produce same tensors'

    def test_seed_numpy(self):
        from src.utils.seed import set_seed

        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


class TestLogger:
    def test_logger_creates(self):
        from src.utils.logger import get_logger
        logger = get_logger('test_module', log_dir='logs')
        assert logger is not None
        assert logger.name == 'test_module'

    def test_logger_writes(self):
        from src.utils.logger import get_logger
        logger = get_logger('test_write', log_dir='logs')
        logger.info('Phase 0 sanity check')
        log_path = os.path.join('logs', 'test_write.log')
        assert os.path.exists(log_path)

    def test_logger_singleton(self):
        from src.utils.logger import get_logger
        l1 = get_logger('singleton_test')
        l2 = get_logger('singleton_test')
        assert l1 is l2


class TestMetrics:
    def test_sharpe_ratio(self):
        from src.utils.metrics import sharpe_ratio
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.008])
        sr = sharpe_ratio(returns)
        assert isinstance(sr, float)
        assert not np.isnan(sr)

    def test_sharpe_zero_std(self):
        from src.utils.metrics import sharpe_ratio
        returns = np.array([0.001, 0.001, 0.001])
        sr = sharpe_ratio(returns)
        assert sr == 0.0  # constant returns -> std=0 -> return 0

    def test_max_drawdown(self):
        from src.utils.metrics import max_drawdown
        # Portfolio goes 100 -> 120 -> 90 -> 110
        values = np.array([100, 120, 90, 110])
        mdd = max_drawdown(values)
        assert mdd < 0  # Should be negative
        assert abs(mdd - (-0.25)) < 0.01  # 90/120 - 1 = -25%

    def test_sortino_ratio(self):
        from src.utils.metrics import sortino_ratio
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        sr = sortino_ratio(returns)
        assert isinstance(sr, float)
        assert not np.isnan(sr)

    def test_calmar_ratio(self):
        from src.utils.metrics import calmar_ratio
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.008])
        values = np.cumprod(1 + returns) * 100
        cr = calmar_ratio(returns, values)
        assert isinstance(cr, float)

    def test_annualized_return(self):
        from src.utils.metrics import annualized_return
        # 248 days of 0.04% daily = ~10.4% annualized
        returns = np.full(248, 0.0004)
        ar = annualized_return(returns)
        assert 0.09 < ar < 0.12

    def test_portfolio_turnover(self):
        from src.utils.metrics import portfolio_turnover
        weights = np.array([
            [0.5, 0.5],
            [0.6, 0.4],
            [0.4, 0.6],
        ])
        to = portfolio_turnover(weights)
        assert to > 0


class TestProjectStructure:
    def test_directories_exist(self):
        dirs = ['src', 'src/data', 'src/sentiment', 'src/graph', 'src/rl',
                'src/gan', 'src/nas', 'src/federated', 'src/quantum',
                'src/api', 'src/utils', 'tests', 'configs', 'data',
                'models', 'experiments', 'dashboard', 'thesis']
        for d in dirs:
            path = os.path.join(PROJECT_ROOT, d)
            assert os.path.isdir(path), f'Missing directory: {d}'

    def test_config_file_exists(self):
        path = os.path.join(PROJECT_ROOT, 'configs', 'base.yaml')
        assert os.path.isfile(path)

    def test_gitignore_exists(self):
        path = os.path.join(PROJECT_ROOT, '.gitignore')
        assert os.path.isfile(path)
