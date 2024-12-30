import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import empirical_covariance
from pypfopt.risk_models import CovarianceShrinkage
import statsmodels.api as sm
import mgarch #最终选择DCC-GARCH部分采用的模块
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class CovarianceBasic():
    """
        协方差估计中用到的最底层的基类
    """
    def __init__(self, stock_returns=None):
        self.stock_returns = stock_returns # stocks*n_days

    def stock_returns_example(self, n_days=504, stocks=5):
        stock_returns = np.random.normal(0.001, 0.02, (n_days, stocks))
        self.stock_returns = stock_returns

    def cal_emp_cov(self,stock_returns = 'auto'):
        """
        :return: 计算历史协方差矩阵
        """
        if stock_returns == 'auto':
            stock_returns = self.stock_returns
        return empirical_covariance(stock_returns, assume_centered=False)

class FactorBasic(CovarianceBasic):
    """
        因子模型的基类
    """
    def __init__(self, factor_matrix=None, stock_returns=None, window_size=252):
        self.factor_matrix = factor_matrix # factors*n_days
        self.window_size = window_size
        super().__init__(stock_returns)

    def calculate_omega_m(self):
        """
        :return:因子协方差矩阵
        """
        recent_factors = self.factor_matrix[-self.window_size:]
        omega_m = np.cov(recent_factors.T, ddof=1)
        if omega_m.ndim == 0:
            return omega_m.item()
        else:
            return omega_m

    def ols_cal(self):
        """
        :return: beta 回归系数矩阵，specific_risk_matrix 特质风险矩阵
        """
        factor_model = LinearRegression()
        factor_model.fit(self.factor_matrix, self.stock_returns)
        beta = factor_model.coef_
        predict_return = factor_model.predict(self.factor_matrix)
        residuals_matrix = self.stock_returns - predict_return
        specific_risk_matrix = np.diag(np.var(residuals_matrix, axis=0,ddof=1))
        return beta, specific_risk_matrix


    def calculate_covariance_matrix(self):
        """
        :return: 协方差估计结果
        """
        beta, specific_risk_matrix = self.ols_cal()
        omega_m = self.calculate_omega_m()
        return self.cal_judge(omega_m,beta) + specific_risk_matrix

    def cal_judge(self,omega,beta):
        if isinstance(omega, float):
            return beta @ beta.T * omega
        else:
            return beta @ omega @ beta.T

    def factor_example(self, n_days=504, factors=3, stocks=5):
        """
        :param n_days: 天数
        :param factors:因子数量
        :param stocks: 股票数量
        :return: 产生因子矩阵和假设的实际收益率数据
        """
        self.factor_matrix = np.random.normal(0.001, 0.02, (n_days,factors))
        # 可以选择特定的生成方式生成测试数据 也可以简化为直接使用随机数据 达到模拟实际数据输入的效果
        # true_beta = np.random.normal(0.001, 0.02, (stocks, factors))
        # stock_returns = true_beta @ self.factor_matrix.T + np.random.normal(0, 0.02, (stocks,n_days))
        # stock_returns = stock_returns.T
        self.stock_returns_example(n_days, stocks)


class BetaEstimator:
    '''用来估计市场模型的beta，包含三种方法'''

    def __init__(self, window_size=252):
        self.window_size = window_size

    def _calculate_rolling_betas(self, stock_returns, market_returns):
        """计算给定窗口内的滚动beta"""
        betas = []
        std_errs = []

        for i in range(len(stock_returns) - self.window_size + 1):
            X = market_returns[i:i + self.window_size]
            y = stock_returns[i:i + self.window_size]
            X = sm.add_constant(X)
            model = sm.OLS(y, X)
            results = model.fit()
            betas.append(results.params[1])
            std_errs.append(results.bse[1])

        return np.array(betas), np.array(std_errs)

    def simple_beta(self, stock_returns, market_returns):
        """无调整beta"""
        betas, _ = self._calculate_rolling_betas(stock_returns, market_returns)
        return betas[-1]  # 返回最新的beta值

    def blume_beta(self, stock_returns, market_returns):
        """blume调整beta"""
        betas, _ = self._calculate_rolling_betas(stock_returns, market_returns)

        if len(betas) < 2:
            raise ValueError("数据量不足，无法进行Blume调整")

        # 使用最后两个时间窗口的beta
        beta_t_2 = betas[-2]
        beta_t_1 = betas[-1]

        # 计算调整系数（这里使用简化的方法：0.67 * 最新beta + 0.33）
        # 这是一个常用的简化版本，假设beta趋向于1
        adjusted_beta = 0.67 * beta_t_1 + 0.33

        return adjusted_beta

    def vasicek_beta(self, stock_returns, market_returns):
        """
        Vasicek调整Beta模型
        """
        # 计算beta值和其标准误
        betas, std_errs = self._calculate_rolling_betas(stock_returns, market_returns)

        if len(betas) == 0:
            raise ValueError("数据量不足，无法计算Beta")

        # 获取最新的beta和其标准误
        current_beta = betas[-1]
        current_std_err = std_errs[-1]

        # 计算beta的横截面均值和方差
        beta_mean = np.mean(betas)
        beta_cross_var = np.var(betas)

        # Vasicek调整
        weight = current_std_err ** 2 / (beta_cross_var + current_std_err ** 2)
        adjusted_beta = weight * beta_mean + (1 - weight) * current_beta

        return adjusted_beta


class MarketBeta(FactorBasic):
    """
        无调整Beta和有调整Beta因子模型
    """
    def __init__(self, factor_matrix=None, stock_returns=None, window_size=252):
        self.beta_estimator = BetaEstimator(window_size)
        super().__init__(factor_matrix, stock_returns, window_size)

    def calculate_specific_risks(self, betas):
        residual_vars = []
        for i, stock in enumerate(self.stock_returns.columns):
            # 获取当前股票的收益率
            stock_returns = self.stock_returns[stock].values[-self.window_size:]
            market_returns = self.factor_matrix[-self.window_size:]

            # 计算回归残差
            predicted_returns = betas[i] * market_returns
            residuals = stock_returns - predicted_returns

            # 计算残差的方差（特质风险）
            residual_var = np.var(residuals, ddof=1)
            residual_vars.append(residual_var)

        # 构建特质风险矩阵 E（对角矩阵）
        specific_risk_matrix = np.diag(residual_vars)
        return specific_risk_matrix

    def calculate_covariance_matrix(self, beta_type='simple'):
        # Step 1: 计算每只股票的 Beta 值
        betas = []
        self.stock_returns = pd.DataFrame(self.stock_returns)
        for stock in self.stock_returns.columns:
            stock_returns = self.stock_returns[stock].values
            if beta_type == 'simple':
                beta = self.beta_estimator.simple_beta(stock_returns, self.factor_matrix)
            elif beta_type == 'blume':
                beta = self.beta_estimator.blume_beta(stock_returns, self.factor_matrix)
            elif beta_type == 'vasicek':
                beta = self.beta_estimator.vasicek_beta(stock_returns, self.factor_matrix)
            else:
                raise ValueError("未知的 Beta 类型")
            betas.append(beta)

        betas = np.array(betas).reshape(-1, 1)  # 转换为列向量

        # Step 2: 计算市场因子的协方差矩阵 Ω_m
        omega_m = self.calculate_omega_m()

        # Step 3: 计算特质风险矩阵 E
        specific_risk_matrix = self.calculate_specific_risks(betas)

        # Step 4: 计算协方差矩阵 Σ
        covariance_matrix = self.cal_judge(omega=omega_m,beta=betas) + specific_risk_matrix
        return covariance_matrix

    def factor_example(self, n_days=504, factors=1, stocks=5):
        super().factor_example(n_days,factors,stocks)
    
class IndustryFactor(FactorBasic):
    """
        行业因子的时间序列回归模型
    """
    def __init__(self, factor_matrix=None, stock_returns=None, window_size=252, industry_matrix=None, mask_matrix=None):
        super().__init__(factor_matrix, stock_returns, window_size)
        self.industry_matrix = industry_matrix
        self.mask_matrix = mask_matrix

    def calculate_omega_ind(self):
        """
        :return:industry因子协方差矩阵
        """
        recent_factors = self.industry_matrix[-self.window_size:]
        omega_ind = np.cov(recent_factors.T, ddof=1)
        if omega_ind.ndim == 0:
            return omega_ind.item()
        else:
            return omega_ind

    def industry_matrix_example(self, stocks=5, industrys=3, n_days=504):
        """
        :param stocks: 股数
        :param industrys: 所有股票涉及的行业数量
        :return: 行业掩码矩阵
        """
        mask_matrix = np.zeros((stocks, industrys), dtype=bool)
        true_indices = np.random.choice(industrys, size=stocks)
        mask_matrix[np.arange(stocks), true_indices] = True
        self.mask_matrix = mask_matrix
        self.industry_matrix = np.random.normal(0.001, 0.02, (n_days, industrys))

    def factor_example(self, n_days=504, factors=1, stocks=5):
        """
        设置参数为使得只有一个市场因子
        """
        super().factor_example(n_days,factors,stocks)

    def industry_regression_step1(self):
        """
        在此时输入的factor_matrix本质上就是市场因子
        :return: 行业残差收益率矩阵
        """
        linear_model1 = LinearRegression()
        linear_model1.fit(self.factor_matrix, self.industry_matrix)
        predict_industry_matrix = linear_model1.predict(self.factor_matrix)
        residuals_matrix = self.industry_matrix - predict_industry_matrix
        return residuals_matrix

    def industry_regression_step2(self):
        """
        分资产做回归计算，过程中利用mask_matrix调整X
        :return: 残差、beta、gamma
        """
        residuals_matrix = self.industry_regression_step1()
        residuals_list = []
        beta_list = []
        for i in range(self.stock_returns.shape[1]):
            temp_factor_matrix = self.factor_matrix.copy()
            temp_factor_matrix = np.hstack((temp_factor_matrix, residuals_matrix*self.mask_matrix[i,:]))
            model1 = LinearRegression()
            model1.fit(temp_factor_matrix, self.stock_returns[:,i])
            beta_list.append(model1.coef_)
            model1_predict = model1.predict(temp_factor_matrix)
            residuals_list.append(self.stock_returns[:,i] - model1_predict)
        residuals_array = np.array(residuals_list).T
        beta_array = np.array(beta_list)
        beta = beta_array[:,0:self.factor_matrix.shape[1]]
        gamma = beta_array[:,self.factor_matrix.shape[1]:]
        return residuals_array, beta, gamma

    def industry_regression_step3(self):
        """
        协方差预测计算
        :return: result
        """
        residuals_array, beta, gamma = self.industry_regression_step2()
        specific_risk_matrix = np.diag(np.var(residuals_array, axis=0, ddof=1))
        omega_m = self.calculate_omega_m()
        omega_ind = self.calculate_omega_ind()
        return self.cal_judge(omega_m,beta) + self.cal_judge(omega_ind,gamma) + specific_risk_matrix

class FactorPCA(FactorBasic):
    """
        在PCA因子模型估计协方差中 原始输入的factor_matrix就是data
    """
    def __init__(self, factor_matrix=None, stock_returns=None, window_size=252):
        super().__init__(factor_matrix, stock_returns, window_size)

    def fpca(self,n='auto',threshold=0.8):
        """
        :param n: 主成分因子个数 默认为'auto' 自动调整
        :param threshold: 在自动调整时用到的阈值
        :return: 主成分和因子载荷矩阵
        """
        data = self.cal_emp_cov()
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data)
        pca = PCA()
        pca.fit(data_standardized)
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(cumulative_variance_ratio >= threshold) + 1
        if isinstance(n,int):
            n_components = n
        elif n == 'auto':
            n_components = num_components
        else:
            raise ValueError(f"{n}值异常")
        pca_final = PCA(n_components=n_components)
        principal_components = pca_final.fit_transform(data_standardized)
        loadings = pca_final.components_.T * np.sqrt(pca_final.explained_variance_)
        return principal_components, loadings

    def factor_pca_main(self):
        """
        :return: pca因子协方差估计结果
        """
        principal_components, loadings = self.fpca()
        specific_risk_matrix = self.cal_emp_cov() - loadings @ principal_components.T
        return self.cal_judge(beta=loadings,omega=np.cov(principal_components.T)) + specific_risk_matrix


class ShrinkageEst(CovarianceBasic):
    """
        这里需要包括两个功能 一个是正常的压缩方差的使用 另一个是目标协方差矩阵是自由定义的（来自于其它的估计结果）
    """
    def __init__(self, stock_returns=None,shrinkage_target="constant_correlation"):
        self.shrinkage_target = shrinkage_target
        super().__init__(stock_returns)

    def shrinkage_cal(self):
        """
        :return: 压缩估计结果
        """
        data = pd.DataFrame(self.stock_returns)
        shrink = CovarianceShrinkage(prices=data, returns_data=True)
        if self.shrinkage_target not in ['constant_variance', 'single_factor', 'constant_correlation']:
            # 这里是如果是天风证券中提到的如果使用其它的自定义的目标协方差矩阵
            # 这里确切完成后 包括天风的8 10 11
            # 无法传入PCA因子法估计的协方差矩阵 天风也没有采用PCA因子法估计得到的协方差矩阵 猜测和PCA得到的协方差矩阵形式有关
            shrink.__dict__['S'] = self.shrinkage_target
            # 这里在替换了目标协方差矩阵后 参考天风证券原文采用2003年据 Lediot & Wolf（2003）继续采用constant_correlation体系下进行估计alpha
            shrinkage_result = shrink.ledoit_wolf(shrinkage_target="constant_correlation")
        else:
            shrinkage_result = shrink.ledoit_wolf(shrinkage_target=self.shrinkage_target)
        return shrinkage_result

class DccGarch(CovarianceBasic):
    def __init__(self, stock_returns=None, window_size=252):
        self.window_size = window_size
        super().__init__(stock_returns)

    def dcc_garch_cal(self):
        """
        :return: DCC_GARCH 模型预测的下一天的协方差矩阵
        """
        stock_returns = self.stock_returns[-self.window_size:]
        vol = mgarch.mgarch()
        vol.fit(stock_returns)
        cov_nextday = vol.predict() #这里是默认后一天
        return cov_nextday['cov']


# class RiskMetrics(CovarianceBasic):
#     """
#         指数移动平均法（EWMA） 在天风证券的研报中这个写为RiskMetrics，原因为RiskMetrics集团
#     """
#     def __init__(self, stock_returns=None, window_size=252):
#         self.window_size = window_size
#         super().__init__(stock_returns)
#
#     def risk_metrics(self, rm_lambda=0.97):
#         stock_returns = self.stock_returns[-self.window_size:]
#         means = stock_returns.mean()
#         epsion = stock_returns - means.reshape(-1, 1)
#         his_cov = self.cal_emp_cov(stock_returns)
#         return (1-rm_lambda)* (epsion @ epsion.T) + rm_lambda * his_cov


# class PortfolioEst():
#     def __init__(self,*args):
#         self.

if __name__ == '__main__':

    """
        已实现协方差估计部分：
        1.基本因子模型
        2.无调整beta市场模型
        3.blume调整beta市场模型
        4.vasicek调整beta市场模型
        5.行业因子模型
        6.PCA因子模型
        7.压缩估计
        8.外部目标协方差矩阵的压缩估计
        9.DCC-GARCH
        
    """

    np.random.seed(0)

    # 基本因子模型
    factor = FactorBasic()
    factor.factor_example(factors=5)
    factor_result = factor.calculate_covariance_matrix()

    # todo 市场模型可以考虑继承自因子基类进行撰写
    # 无调整beta市场模型
    simple_beta = MarketBeta()
    simple_beta.factor_example()
    simple_beta_result = simple_beta.calculate_covariance_matrix()

    # blume调整beta市场模型
    blume_beta = MarketBeta()
    blume_beta.factor_example()
    blume_beta_result = blume_beta.calculate_covariance_matrix(beta_type='blume')

    # vasicek调整beta市场模型
    vasicek_beta = MarketBeta()
    vasicek_beta.factor_example()
    vasicek_beta_result = vasicek_beta.calculate_covariance_matrix(beta_type='vasicek')

    # 行业因子模型
    industry = IndustryFactor()
    industry.factor_example()
    industry.industry_matrix_example()
    industry_factor_result = industry.industry_regression_step3()

    # PCA因子模型
    factor_pca = FactorPCA()
    factor_pca.stock_returns_example()
    factor_pca_result = factor_pca.factor_pca_main()

    # 压缩估计
    shrinkage = ShrinkageEst()
    shrinkage.stock_returns_example()
    shrinkage_result = shrinkage.shrinkage_cal()

    # 外部目标协方差矩阵的压缩估计
    shrinkage_self_target = ShrinkageEst(shrinkage_target=simple_beta_result)
    shrinkage_self_target.stock_returns_example()
    shrinkage_self_target_result = shrinkage_self_target.shrinkage_cal()

    # DCC-GARCH
    dcc_garch = DccGarch()
    dcc_garch.stock_returns_example()
    dccgarch_result = dcc_garch.dcc_garch_cal()

    #


    print(1)



    # todo list
    """
        1.讨论横截面回归的因子模型 （先不做）
        # 2.压缩估计中目标协方差矩阵的更换后测试
        # 3.DCC-GARCH模型部分
        4.RiskMetrics
        5.
    """