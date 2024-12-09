# ## 动态因子模型实现

# ### 原作者
# - 姓名: Gene Kindberg-Hanlon
# - 联系方式: genekindberg @ googlemail .com
# - 原github仓库地址：https://github.com/genekindberg/DFM-Nowcaster.git

# ### 修改内容
# - 本仓库基于 Gene Kindberg-Hanlon 的原始实现进行了以下修改：
# - 修正了原代码在吉布斯采样和BVAR等过程中使用了未来信息的错误
# - 添加了一些中文注释

# ### 许可协议
# - 本代码的共享和使用需遵循原始代码的许可协议（如适用）。



##################################################################
# 导入包
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime as dt
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

##################################################################

def _InterpolateNaN(Tseries): # 线性插值
    not_nan, indices = np.logical_not(np.isnan(Tseries)), np.arange(len(Tseries)).reshape(-1,1)
    Tseries_fill = np.interp(indices, indices[not_nan], Tseries[not_nan]) 
    return Tseries_fill

def _NormDframe(df): # 对输入进行标准化
    df2 = df.copy()
    df2 = (df2 - df2.mean())/df2.std()
    return df2

def _PCA_fact_load(data): # pca，保留一个因子
    pca = PCA(n_components=1)
    factor = pca.fit_transform(data)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    return factor, loadings # 因子，因子载荷

def _MakeLags(mat, lags): # 创建滞后序列
    cols = mat.shape[1]
    Matlag = np.zeros((len(mat)-lags,cols*lags))
    for ii in range(lags, 0, -1):
        Matlag[:,0+cols*(lags-ii):cols*(lags-ii+1)] = mat[ii-1:(len(mat)-lags+ii-1),:];
    mat2 = mat.copy()
    mat2 = mat2[lags:,:]
    return mat2, Matlag 
    
def _estVAR(Yinput,lags):
    # 估计VAR
    # Y：X1t, X2t, X3t, X4t：4*1
    # X: X1t-1, X2t-1, X3t-1, X4t-1, X1t-2, X2t-2, X3t-2, X4t-2……：24*1
    Y, X = _MakeLags(Yinput, lags)
    N = Y.shape[0]
    X = np.c_[X, np.ones((X.shape[0],1))]
    B = np.linalg.inv(X.T @ X) @ (X.T @ Y)
    e = Y - (X @ B)
    Q = np.divide((e.T @ e),(N-lags)) # 4*4，VAR的估计误差
    return B, Q # B：4*24， BX=Y



def _setVARpriors(X,Y,lambda1, lambda3, lambda4, p, alpha0):
    n = Y.shape[1] # 4
    T = X.shape[0] # 300
    k = X.shape[1] # 24
    m = k-n*p # 0
    q=n*k # 96
    arvar = np.empty((n,1))
    arvar[:] = np.nan
    ar = np.empty((n,1))
    ar[:] = np.nan
    for ii in range(0,n,1):
        # initiate the data
        Ytemp=Y[:,ii]
        Xtemp=X[:,ii:n*p:n] # Get heterogenous constants in too
        # obtain the OLS estimator
        B= np.linalg.inv(Xtemp.T @ Xtemp) @ (Xtemp.T @ Ytemp)
        ar[ii] = B[0] # 1st AR term for each equation
        # obtain the vector of residuals
        eps=Ytemp - (Xtemp @ B)
        arvar[ii]=np.divide((eps.T @ eps),(T-p)) # 根据残差构建其先验协方差

    # 上面的代码对于4个变量建立了AR(6)模型，生成了系数，而ar仅存储了该系数的其中一个，也就是
    # 最近一期的系数，所以ar有4个元素，arvar存储了标准误，也是4个元素
    
    
    beta0=np.zeros((q,1));
    for ii in range(0,n,1):
        beta0[(ii)*k+ii]=ar[ii] # 这里的beta0是96*1，但后面实际应用时，还是会将其转为24*4，对每一列，分别第1、2、3、4个元素是上面的B[0]
    
    # variance cov of prior on beta
    phi0=np.zeros((k,k)) # 24*24 注意这是系数的先验协方差矩阵，后面可以看到它衍生自方程的残差S0，但其实仅仅是与其有关系
    
    for ii in range(0,n,1):
       for jj in range(0,p,1):
           # 如前所述，根据S0构建先验，这里是明尼苏达先验构造方法
           phi0[(jj)*n+ii,(jj)*n+ii]=(1/arvar[ii])*((lambda1/((jj+1)**lambda3))**2) 
    
    # 额外的变量的处理
    if k>n*p:
        m = k-n*p
        for ii in range(0,m,1):
            phi0[k-m+ii,k-m+ii]=(lambda1*lambda4)^2      

    S0=np.diag(arvar.flatten()); # 4*4 残差的先验方差
    return phi0, alpha0, S0, beta0


def _EstBVAR(Y,lags, phi0, alpha0, S0, beta0, draws):
    # Y = Data
    # Lags: Number of lags in BVAR
    # phi: variance on the priors of beta

    Y2, X = _MakeLags(Y, lags)
    T = X.shape[0] # time periods
    n = Y2.shape[1] # number of equations
    k = X.shape[1] # parameters in each equation
    
    
    B0=np.reshape(beta0,(k,n), order="F")
    invphi0=np.diag(1./np.diag(phi0))

    # compute phibar, defined in (1.4.16)
    invphibar=invphi0+(X.T @ X)
    C=np.linalg.cholesky(invphibar).T # Make upper triangular cholesky
    invC=np.linalg.inv(C)
    phibar=invC @ invC.T
    
    # Get ex post B matrix
    Bbar=phibar @ ((invphi0 @ B0) + (X.T @ Y2)) 
    
    # vectorise 
    betabar=Bbar.flatten().T
    
    # obtain alphabar, defined in (1.4.18)
    alphabar=T+alpha0
    
    
    # obtain ex post scale matrix S
    Sbar=(Y2.T @ Y2)+S0+(B0.T @ invphi0 @ B0)-(Bbar.T @ invphibar @ Bbar)

    
    # Initilize Gibbs draw
    sigma_gibbs = np.full((draws,n,n),np.nan) 
    beta_gibbs =np.full((draws,k,n),np.nan) 

    for ii in range(0,draws):
        # Draw Sigma
        C=np.linalg.cholesky(Sbar)
        
        # draw the matrix Z of alpha multivariate standard normal vectors
        Z = np.random.multivariate_normal(np.full(n,0), np.diag(np.full(n,1)), alphabar)
        # Adjust for var cov
        drawSig=(C @ np.linalg.inv(Z.T @ Z)) @ C.T
        sigma_gibbs[ii,:,:]=drawSig # store draw
        
        
        # Draw beta
        # compute the lower Choleski factor of sigma draw
        C=np.linalg.cholesky(drawSig)
        
        # compute the lower choleski factor of phi
        P=np.linalg.cholesky(phibar)
        
        # take a kn*1 random draw from a multivariate standard normal distribution, and redimension it to obtain a k*n matrix-variate normal
        w=np.random.normal(0, 1, k*n)
        # reshape to be shape of the beta mat
        W=np.reshape(w,(k,n))
        
        # obtain the random draw from the matrix-variate student by adding the location matrix and multiplying by both scale matrices
        drawBeta=Bbar+((P @ W) @ C.T)
        drawB = np.r_[drawBeta[0:,0:].T, np.c_[np.eye((n)*(lags-1)), np.zeros(((n)*(lags-1),(n)))]]

        #Check stability and redraw if necessary - break if not under stability limit
        count = 1
        while np.max(np.abs(np.linalg.eigvals(drawB)))>0.999:
            W = np.reshape(np.random.normal(0, 1, k*n),(k,n))
            drawBeta=Bbar+((P @ W) @ C.T)
            drawB = np.r_[drawBeta[0:,0:].T, np.c_[np.eye((n)*(lags-1)), np.zeros(((n)*(lags-1),(n)))]]
            count = count+1
            if count>10000:
                raise Exception('VAR not stationary!')
        
        beta_gibbs[ii,:,:]=drawBeta    
    return sigma_gibbs, beta_gibbs


def _Kfilter(Y, F, H, Q, R, F0): # F0为初始状态向量
    k=F0.shape[1] # 状态向量维度
    T = Y.shape[0] # 时间长度
    P = np.full((T,k**2),np.nan) # 用P存储协方差矩阵，但注意该矩阵应为K*K维度
    S = np.full((T,k),np.nan)  # 用S存储状态估计
    Sp = F0[0,:] 
    Pp=np.eye(k)
    
    for i in range(0,T,1):
        y = Y[i,:].T.copy() # Y为观测矩阵，y为Y的第i个时间点，也即第i行
        
        if np.isnan(y).any():
            replace = H @ Sp # 如果该行有任何缺失值，用预测值代替
            y[np.where(np.isnan(y))] = replace[np.where(np.isnan(y))] 
        
        nu = y - (H @ Sp)   # 计算估计误差
        f = ((H @ Pp) @ H.T) + R # f和finv用来更新卡尔曼增益
        finv=np.linalg.inv(f)
        
        Stt = Sp + (Pp @ H.T) @ (finv @ nu) # (Pp @ H.T) @ (finv)为更新的卡尔曼增益，这一步是状态更新
        Ptt = Pp - ((Pp @ H.T) @ (finv @ H) @ Pp) # 误差协方差更新
        
        if i < T:
            Sp = F @ Stt # 状态外推，或者说下一步的状态预测
            Pp = ((F @ Ptt) @ F.T) + Q # 误差协方差外推
        
        
        S[i,:] = Stt.T # 存储这一时间点的状态
        P[i,:] = np.reshape(Ptt,(k**2)) # 存储这一时间点的误差协方差矩阵
        
    return S, P # 返回状态估计和协方差矩阵


def _Ksmoother(F, H, Q, R, S, P, lags, n):
    # F: Transition matrix
    # H: Observation loadings
    # Q: Transition variance
    # R: Observation variance
    # S: Kalman filtered states
    T = S.shape[0]
    k = n*lags
    Qstar=Q[0:n,0:n]
    Fstar=F[0:n,:]
    Pdraw = np.full((T,k**2),np.nan) 
    Sdraw = np.full((T,k),np.nan)
    for ii in range(T-1,-1,-1):
        Sf = S[ii,0:n].T
        Stt = S[ii-1,:]
        Ptt = np.reshape(P[ii-1,:],(k,k))
        f = (Fstar @ Ptt @ Fstar.T) + Qstar
        finv = np.linalg.inv(f);
        nu = Sf - (Fstar @ Stt)
        
        Smean = Stt + (Ptt @ Fstar.T  @ finv @ nu)
        Svar = Ptt - (Ptt @ Fstar.T @ finv @ Fstar @ Ptt)
       
        Sdraw[ii,:] = Smean.T
        Pdraw[ii,:] = np.reshape(Svar,(k**2))           
    
    return Sdraw, Pdraw, Svar

def _RemoveOutliers(Data, SDs, Qs):
    for col in range(0,Data.shape[1]):
        Data[(Data[:,col] - np.nanmean(Data[:,col])) > SDs * np.nanstd(Data[:,col]), col] =  np.nanmean(Data[:,col])+SDs * np.nanstd(Data[:,col])
        Data[(Data[:,col] - np.nanmean(Data[:,col])) < -(SDs * np.nanstd(Data[:,col])), col] =  np.nanmean(Data[:,col])-SDs * np.nanstd(Data[:,col])
    return Data

def _SampleFactorsSimple(S,Svar, K, Qs):
    keep = np.setdiff1d(np.arange(0,S.shape[1]), np.arange(0,S.shape[1],(K+Qs))) # miss first so dont sample Quarterly
    Sdraw = S.copy()
    for ii in range(S.shape[0]):
        Sdraw[ii,keep] = np.random.multivariate_normal(S[ii,keep].T, np.diag(np.diag(Svar)[keep]), 1)

    return Sdraw

def _SampleLoadingsLags(S, XY, s0, alpha0, L_var_prior, lags, interv, Qs, K, lagsH):
    
    # S: Estimated factors, including lags
    # s0: Scale param for R
    # alpha0: deg of freedom param for R
    # b0: prior on loading coefficients
    # B0: prior variance on loading coefficients.
    # lags: number of lags of monthly data in DFM.
    # interv. Intervals of the data for each factor Y
    # Qs: Number of quarterly observable GDP series
    # Monthly_dat: matrix of all monthly data
    K = int(S.shape[1]/lags-Qs)
    T = XY.shape[0]
    N = XY.shape[1]
    # Get rid of nan
    XYnan = XY[~np.isnan(XY[:,Qs:]).any(axis=1)].copy()
    Beta = []
    cumsum = 0
    for ii in range(0,len(interv)):
        Beta.append( np.linalg.inv(S[0:XYnan.shape[0],Qs+ii:Qs+ii+(lagsH-1)*(Qs+K)+1:Qs+K].T @ S[0:XYnan.shape[0],Qs+ii:Qs+ii+(lagsH-1)*(Qs+K)+1:Qs+K]) @ (S[0:XYnan.shape[0],Qs+ii:Qs+ii+(lagsH-1)*(Qs+K)+1:Qs+K].T @ XYnan[:,Qs+cumsum:Qs+cumsum+interv[ii]]))
        cumsum = cumsum+Beta[ii-Qs].shape[1]

    Lf = np.zeros((N-Qs,K*lagsH+Qs*lagsH))
    # Add beta loading for each lag - add zeros for loading onto quarterly series
    for jj in range(0,lagsH):
        cumsize = 0
        for ii in range(len(interv)+Qs): # Make matrix of factor loadings for each factor
            if ii==0 or ii+1 % (K+Qs)==0:
                Lf[:,ii:ii+1] = np.zeros((N-Qs,1))
            else:
                Lf[cumsize:cumsize+Beta[ii-Qs].shape[1],jj*(K+Qs)+ii] = Beta[ii-Qs][jj,:].reshape(-1,1).T # Beta val on current factor
                # -phi*beta on lagged factor, and also b(2) if the data loads on to multiple factor lags
                cumsize = cumsize+Beta[ii-Qs].shape[1] 

    # Make loading matrix for observations including quarterly data
    Lf_con = np.c_[np.tile(np.c_[1/3, np.zeros((1,K))],(1,3)), np.tile(np.zeros((1,K+Qs)),(1,lags-3))]
    Lf = np.r_[Lf_con, np.c_[Lf, np.zeros((N-Qs,(lags-lagsH)*(K+Qs)))]]
    # Make bayesian draws of variance matrix for factor loading using 
    StoreE = np.empty([XY.shape[0],XY.shape[1]]) 
    R = np.zeros((XY.shape[1],XY.shape[1]))
    for n in range(0,XY.shape[1]):
        ed=XY[:,n] - S[:,:] @ Lf[n,:].T
        StoreE[:,n] = ed
        ed = ed[~np.isnan(ed)]
        # draw R(n,n)
        if n==0:
            s0=0.01
        else:
            s0=5
        R_bar = s0 +(ed.T @ ed)+Lf[n,:] @ np.linalg.inv(L_var_prior+np.linalg.inv(S.T @ S)) @ Lf[n,:].T # Assuming prior of zero, so no L          
        Rd = np.random.normal(0,1,T+alpha0)
        Rd = Rd.T @ Rd
        Rd = np.divide(R_bar,Rd)
        R[n,n]=Rd
    H = Lf
    return H, R, StoreE

def _SampleLoadingsLagsPhi(S, XY, s0, alpha0, L_var_prior, lags, interv, Qs, K, lagsH, Phi):
    
    # S: Estimated factors, including lags
    # s0: Scale param for R
    # alpha0: deg of freedom param for R
    # b0: prior on loading coefficients
    # B0: prior variance on loading coefficients.
    # lags: number of lags of monthly data in DFM.
    # interv. Intervals of the data for each factor Y
    # Qs: Number of quarterly observable GDP series
    # Monthly_dat: matrix of all monthly data
    K = int(S.shape[1]/lags-Qs)
    T = XY.shape[0]
    N = XY.shape[1]

    # Gen Quasi-differenced data
    Y_phi = _quasidiff(XY,Phi, Qs)
    Beta = []
    cumsum = 0
    # Get relevant factor for each observable and quasi difference the factor by the relevant phi, then store the estimated beta with the same block
    cumsize = 0
    for ii in range(0,len(interv)):
        Betatemp = np.zeros((lagsH,interv[ii]))
        for jj in range(0,interv[ii]):
            # Each lag uses the same quasidiff so it is tiled lagsH times
            notnan = ~np.isnan(Y_phi[:,Qs:]).any(axis=1)
            Sreg = _quasidiff(S[notnan,Qs+ii:Qs+ii+(lagsH)*(Qs+K):Qs+K],np.tile(Phi[cumsize+jj,:],(lagsH,1)),0)
            notnan[0] = False # exclude first obs given that S has been quasi differenced 
            Best = np.linalg.inv(Sreg.T @ Sreg) @ (Sreg.T @ Y_phi[notnan,Qs+cumsum+jj:Qs+cumsum+jj+1])
            Betatemp[:,jj] = Best.reshape((1,-1))
        Beta.append(Betatemp)
        cumsize = cumsize+interv[ii]
            
            
    Lf = np.zeros((N-Qs,K*(lagsH+1)+Qs*(lagsH+1))) # Need one extra factor above lagsH because of MA term
    Lfraw = np.zeros((N-Qs,K*(lagsH)+Qs*(lagsH))) # Loading matrix without phi - used to generate errors with which to sample phi 
    # Add beta loading for each lag - add zeros for loading onto quarterly series (since there is no MA term here)
    for jj in range(0,lagsH):
        cumsize = 0
        for ii in range(K+Qs): # Make matrix of factor loadings for each factor
            if ii==0 or ii+1 % (K+Qs)==0:
                Lf[:,ii:ii+1] = np.zeros((N-Qs,1))
            else:
                # Add beta to existing entry (may already be a -phi*beta in that location)
                Lf[cumsize:cumsize+Beta[ii-Qs].shape[1],jj*(K+Qs)+ii] = Lf[cumsize:cumsize+Beta[ii-Qs].shape[1],jj*(K+Qs)+ii]+Beta[ii-Qs][jj,:].reshape(-1,1).T
                Lfraw[cumsize:cumsize+Beta[ii-Qs].shape[1],jj*(K+Qs)+ii] = Beta[ii-Qs][jj,:].reshape(-1,1).T
                Lf[cumsize:cumsize+Beta[ii-Qs].shape[1],(jj+1)*(K+Qs)+ii] = (-Phi[cumsize:cumsize+Beta[ii-Qs].shape[1],:].T)*Beta[ii-Qs][jj,:].reshape(1,-1)
                cumsize = cumsize+Beta[ii-Qs].shape[1] 

    # Make loading matrix for observations including quarterly data
    Lf_con = np.c_[np.tile(np.c_[1/3, np.zeros((1,K))],(1,3)), np.tile(np.zeros((1,K+Qs)),(1,lags-3))]
    Lf = np.r_[Lf_con, np.c_[Lf, np.zeros((N-Qs,(lags-lagsH-1)*(K+Qs)))]] # -1 for MA term 
    Lfraw = np.r_[Lf_con, np.c_[Lfraw, np.zeros((N-Qs,(lags-lagsH)*(K+Qs)))]]
    # Make bayesian draws of variance matrix for factor loading using 
    StoreE = np.empty([Y_phi.shape[0],Y_phi.shape[1]]) 
    R = np.zeros((XY.shape[1],XY.shape[1]))
    for n in range(0,Y_phi.shape[1]):
        ed=Y_phi[:,n] - S[:,:] @ Lf[n,:].T
        edraw = XY[1:,n]-S[:,:] @ Lfraw[n,:].T
        StoreE[:,n] = edraw
        ed = ed[~np.isnan(ed)]
        # draw R(n,n)
        if n==0:
            s02=s0/10
        else:
            s02=s0*5
        R_bar = s02 +(ed.T @ ed)+Lf[n,:] @ np.linalg.inv(L_var_prior+np.linalg.inv(S.T @ S)) @ Lf[n,:].T # Assuming prior of zero, so no L          
        # for scale

        Rd = np.random.normal(0,1,T+alpha0)
        Rd = Rd.T @ Rd
        Rd = np.divide(R_bar,Rd)

        R[n,n]=Rd
    H = Lf
    return H, R, StoreE

def _SamplePhi(StoreE,R, Qs):
    Phi = np.zeros((StoreE.shape[1]-Qs,1))
    for ii in range(Qs,StoreE[:,Qs:].shape[1]):
        eps = StoreE[~np.isnan(StoreE[:,ii]),ii].reshape(-1,1)
        Eps, EpsL = _MakeLags(eps, 1)
        Phi[ii-1,0] = np.linalg.inv(EpsL.T @ EpsL) @ (EpsL.T @ Eps)

    return Phi 

def _quasidiff(Y,Phi,Skip):


    Yphi = Y[1:,Skip:]-Y[0:-1,Skip:] * Phi.T
    if Skip>0:
        Yphi = np.c_[Y[1:,0:Skip],Yphi]

    return Yphi



def initializevals(start, GDP, Monthly, lags, lagsH, K, Qs):
    # 截取到 start 时间点之前的数据
    cutoff = int(np.where(GDP['Date'] <= start)[0][-1] + 1) * 3  # 截取到 start 时间点对应的季度末
    GDPnan = np.asarray(np.kron(GDP[:cutoff // 3]['GDP'].values.reshape(-1,1), [[np.nan], [np.nan], [1]]))  # 填充季度数据为月度
    GDP_fill = _InterpolateNaN(GDPnan)  # 插值填充缺失值
    
    # 标准化输入数据
    MonthlyDat_norm = []
    for df in Monthly:
        MonthlyDat_norm.append(_NormDframe(df[:cutoff]))  # 截取并标准化每个数据框

    loadings = [None] * len(Monthly)  # 列表，长度为月度数据的数量
    Factors = [None] * len(Monthly)  # 列表，长度为月度数据的数量
    # 获取主成分
    for df, num in zip(MonthlyDat_norm, range(len(MonthlyDat_norm))):
        Factors[num], loadings[num] = _PCA_fact_load(df.ffill())  # 填充缺失值并计算主成分载荷
    
    # Factors中元素：cutoff*1； loadings中元素：变量数*1

    ## 为因子载荷矩阵和因子生成初始猜测
    F0 = np.concatenate(Factors, 1)  # F0：cutoff * K，包含 K 个因子的矩阵
    N = len(np.concatenate(loadings, 0))  # 因子载荷总数
    Lf = np.zeros((N, len(loadings)))  # N * K
    cumsize = 0 
    for ii in range(len(loadings)):  # 遍历每个月度数据
        Lf[cumsize:cumsize + len(loadings[ii]), ii] = loadings[ii].reshape(1, -1)
        cumsize = cumsize + len(loadings[ii])

    # 创建因子初始化的滞后矩阵，包括插值后的月度 GDP
    F0 = np.hstack((GDP_fill, F0))  # cutoff * (Qs + K)

    F0_cut, F0lag = _MakeLags(F0, lags)  # 创建滞后变量

    # 为观测值创建载荷矩阵
    Lf_con = np.c_[np.tile(np.c_[1 / 3, np.zeros((1, K))], (1, 3)), np.tile(np.zeros((1, K + Qs)), (1, lags - 3))] 
    H = np.r_[Lf_con, np.c_[np.zeros((N, 1)), Lf, np.zeros((N, (lags - 1) * (K + 1)))]]  # 观测矩阵

    Monthly_dat = np.concatenate(MonthlyDat_norm, axis=1)  # cutoff * N， T * 自变量数量

    ###### 获取 R，初始猜测的观测矩阵的方差：
    e = np.c_[GDP_fill[lags:, 0:], Monthly_dat[lags:, 0:]] - F0lag @ H.T  # 计算误差
    e = e[~np.isnan(e).any(axis=1)]  # 去掉含有 NaN 的行
    R = np.divide((e.T @ e), (Monthly_dat.shape[0] - lags))  # 标准化误差
    R = np.diag(np.diag(R))  # 对角化，假设观测变量之间独立

    ### 获取 VAR 的初始参数和方差 
    B, Q = _estVAR(F0_cut, lags)

    Q = np.c_[Q, np.zeros((K + 1, (K + 1) * (lags - 1)))]  # 适应卡尔曼格式
    Q = np.r_[Q, np.zeros(((K + 1) * (lags - 1), (K + 1) * (lags)))] 
    F = np.r_[B[0:-1, 0:].T, np.c_[np.eye((K + Qs) * (lags - 1)), np.zeros(((K + Qs) * (lags - 1), (K + Qs)))]]

    # 因子估计的数据 - 每3个月的 GDP 和月度标准化数据系列
    XY = np.c_[GDPnan, Monthly_dat]
    S, P = _Kfilter(XY, F, H, Q, R, F0lag)
    
    ints = [listel.shape[0] for listel in loadings]  # 每个因子上可观测的载荷数量

    return XY, H, R, F, Q, S, ints



def Gibbs_loop(XY, F, H, Q, R, S, lags, lagsH, K, Qs, s0, alpha0, L_var_prior, Ints, burn, save, GDPnorm):
    # XY: 月度数据 306 * 7
    # F: 状态转移初始矩阵 24 * 24
    # H: 观测初始矩阵 7 * 24
    # Q: 状态转移方差项 24 * 24
    # R: 加载项方差项 7 * 7
    # S: 状态初始矩阵 306 * 24
    # lags: 状态转移矩阵中的滞后阶数 6
    # lagsH: 加载项矩阵中的滞后阶数
    # K: 月度"因子"数量 3
    # Qs: 季度因子数量 1
    # s0: 加载项方差的先验 7 * 7
    # alpha0: 加载项方差的缩放参数
    # L_var_prior: 加载项系数的先验方差（默认为0）
    # Ints: 加载项的限制 - 每个因子加载的数量
    # GDPnorm: 是否标准化GDP
    # 创建存储结果的矩阵
    
    Hdraw = np.empty([H.shape[0], Q.shape[1], save])  # 存储H的抽样结果
    Qdraw = np.empty([Q.shape[0], Q.shape[1], save])  # 存储Q的抽样结果
    Fdraw = np.empty([F.shape[0], F.shape[1], save])  # 存储F的抽样结果
    Pdraw = np.empty([S.shape[0], S.shape[1]*S.shape[1], save])  # 存储P的抽样结果
    Rdraw = np.empty([R.shape[0], R.shape[1], save])  # 存储R的抽样结果
    Sdraw = np.empty([S.shape[0], S.shape[1], save])  # 存储S的抽样结果

    iter = burn + save  # 总迭代次数
    n = K + Qs  # 因子总数

    # 初始化状态转移的先验
    lambda1 = 0.5
    lambda3 = 1
    lambda4 = 1000
    n = (K + Qs) # 4
    alpha0 = 1 
    S_Y, S_X = _MakeLags(S[:, 0:K+Qs], lags)  # S：306 * 24，初始状态估计
    phi0, alpha0, S0, beta0 = _setVARpriors(S_X, S_Y, lambda1, lambda3, lambda4, lags, alpha0)  # 设置VAR先验
    keep = np.setdiff1d(np.arange(0, S.shape[1]), np.arange(0, S.shape[1], n))  # 保留非GDP因子
    for ii in range(iter):
        if ii % 10 == 0:
            print('已完成 ', str(ii/iter*100), "% 的循环")
        S, P = _Kfilter(XY, F, H, Q, R, S)  # 卡尔曼滤波
        S, P, Svar = _Ksmoother(F, H, Q, R, S, P, lags, n)  # 卡尔曼平滑
        # 重新采样S并加入随机冲击
        S = _SampleFactorsSimple(S, Svar, K, Qs)  # 简单采样因子

        # 标准化状态（除了GDP）
        if GDPnorm:
            S[:, :] = (S[:, :] - np.mean(S[:, :], axis=0)) / np.std(S[:, :], axis=0)
        else:
            S[:, keep] = (S[:, keep] - np.mean(S[:, keep], axis=0)) / np.std(S[:, keep], axis=0)

        draws = 1
        sigma_gibbs, beta_gibbs = _EstBVAR(S[:, 0:K+Qs], lags, phi0, alpha0, S0, beta0, draws)  # 估计BVAR
        sigma = sigma_gibbs.reshape((sigma_gibbs.shape[1], sigma_gibbs.shape[2]))
        beta = beta_gibbs.reshape((beta_gibbs.shape[1], beta_gibbs.shape[2]))
        # 更新方差和状态转移矩阵
        Q = np.c_[sigma, np.zeros((K+1, (K+1)*(lags-1)))]  # 适应卡尔曼滤波格式
        Q = np.r_[Q, np.zeros(((K+1)*(lags-1), (K+1)*(lags)))]
        F = np.r_[beta[0:, 0:].T, np.c_[np.eye((K+Qs)*(lags-1)), np.zeros(((K+Qs)*(lags-1), (K+Qs)))]]
        
        # 创建加载项的区间，以便明确哪些因子加载到哪些变量上
        H, R, StoreE = _SampleLoadingsLags(S, XY, s0, alpha0, L_var_prior, lags, Ints, Qs, K, lagsH)

        if ii >= burn:
            Hdraw[:, :, ii-burn] = H.copy()
            Qdraw[:, :, ii-burn] = Q.copy()
            Fdraw[:, :, ii-burn] = F.copy()
            Pdraw[:, :, ii-burn] = P.copy()
            Rdraw[:, :, ii-burn] = R.copy()
            Sdraw[:, :, ii-burn] = S.copy()
                
    return Hdraw, Qdraw, Fdraw, Pdraw, Rdraw, Sdraw  # 返回抽样结果


def Gibbs_loopMA(XY,F, H, Q, R, S, lags, lagsH, K, Qs, s0, alpha0, L_var_prior, Ints, burn, save, GDPnorm):
    # XY: Monthly data
    # F: State transition initial
    # H: Loadings initial mat
    # Q: State transition variance term
    # R: Loading variance term
    # S: States initial
    # lags: number of lags in transition matrix
    # n: length of monthly
    # K: monthly factors
    # Qs: quarterly factors
    # s0: prior on factor loading variance
    # alpha0: scale param on factor loading variance:
    # L_var_prior: tightness on coefficient in factor loading (zero is prior)
    # Ints: Restrictions for loading on factors - number that load onto each
    # GDPnorm: normalize GDP factor?

    # Create matrices to store results
    
    Hdraw = np.empty([H.shape[0],Q.shape[1],save]) 
    Qdraw = np.empty([Q.shape[0],Q.shape[1],save]) 
    Fdraw = np.empty([F.shape[0],F.shape[1],save])
    Pdraw = np.empty([S.shape[0]-1,S.shape[1]*S.shape[1],save]) # reduced by one due to MA term
    Rdraw = np.empty([R.shape[0],R.shape[1],save])
    Sdraw = np.empty([S.shape[0]-1,S.shape[1],save]) # reduced by one due to MA term
    Phidraw = np.empty([XY.shape[1]-Qs,1,save]) # reduced by one due to MA term

    iter = burn+save
    n = K+Qs

    #Initilize priors for state transition 
    lambda1 = 0.5
    lambda3 = 1
    lambda4 = 1000
    n = (K+Qs)
    alpha0=1
    S_Y, S_X = _MakeLags(S[:,0:K+Qs], lags)
    phi0, alpha0, S0, beta0 = _setVARpriors(S_X,S_Y,lambda1, lambda3, lambda4, lags, alpha0)
    Phi = np.zeros((XY.shape[1]-Qs,1)) # Initialize MA coef at 0
    keep = np.setdiff1d(np.arange(0,S.shape[1]), np.arange(0,S.shape[1],n))
    for ii in range(iter):
        if ii % 10==0:
            print('Completed ', str(ii/iter*100), "% of the loop")

        XYquas = _quasidiff(XY,Phi, Qs)
        S, P  = _Kfilter(XYquas, F, H, Q, R, S)
        S, P, Svar = _Ksmoother(F, H, Q, R, S, P, lags, n)
        # Resample S with random shocks
        S = _SampleFactorsSimple(S,Svar, K, Qs)

        #Normalize states (except GDP if GDPnorm switched off)
        if GDPnorm:
            S[:,:] = (S[:,:]-np.mean(S[:,:], axis=0))/np.std(S[:,:],axis=0)
        else:
            S[:,keep] = (S[:,keep]-np.mean(S[:,keep], axis=0))/np.std(S[:,keep],axis=0)


        draws = 1
        sigma_gibbs, beta_gibbs = _EstBVAR(S[:,0:K+Qs],lags, phi0, alpha0, S0, beta0, draws)
        sigma = sigma_gibbs.reshape((sigma_gibbs.shape[1],sigma_gibbs.shape[2]))
        beta = beta_gibbs.reshape((beta_gibbs.shape[1],beta_gibbs.shape[2]))
        # Update Variance and State transition matrices
        Q = np.c_[sigma, np.zeros((K+1,(K+1)*(lags-1)))] # adapt to kalman formats
        Q = np.r_[Q, np.zeros(((K+1)*(lags-1),(K+1)*(lags)))] 
        F=np.r_[beta[0:,0:].T, np.c_[np.eye((K+Qs)*(lags-1)), np.zeros(((K+Qs)*(lags-1),(K+Qs)))]]
        
        # Make intervals for factor loadings so its clear which loads onto which
    
        H, R, StoreE = _SampleLoadingsLagsPhi(S, XY, s0, alpha0, L_var_prior, lags, Ints, Qs, K, lagsH, Phi)
        Phi = _SamplePhi(StoreE,R, Qs)

        if ii >= burn:
            Hdraw[:,:,ii-burn] = H.copy()
            Qdraw[:,:,ii-burn] = Q.copy()
            Fdraw[:,:,ii-burn] = F.copy()
            Pdraw[:,:,ii-burn] = P.copy()
            Rdraw[:,:,ii-burn] = R.copy()
            Sdraw[:,:,ii-burn] = S.copy()
            Phidraw[:,:,ii-burn] = Phi.copy()
                
    return Hdraw, Qdraw, Fdraw, Pdraw, Rdraw, Sdraw, Phidraw 

#############################################################################
#  Dynamic factor model class - includes initialization ,estimation, and nowcasting routines

class DynamicFactorModel():

    def __init__(self, start, Q_GDP, Monthly, Dates, K, Qs, lags, lagsH, MAterm, normGDP):
        self.GDP = Q_GDP # GDP数据，季频
        self.Monthly = Monthly # 其他数据，月频
        self.Dates = Dates
        self.K = K # 因子数目
        self.Qs = Qs # 因变量的数目，本代码Qs为1，即GDP
        self.lags = lags # 状态转移方程滞后阶数
        self.lagsH = lagsH
        self.MAterm = MAterm
        self.normGDP = normGDP
        self.time = start

        self.XY, self.H, self.R, self.F, self.Q, self.S, self.ints = initializevals(
                                                                     self.time, self.GDP, self.Monthly, self.lags, self.lagsH, self.K, self.Qs)
        
    def estimateGibbs(self, F, H, Q, R, S, ints,
                            burn, save, s0 = 0.1, alpha0 = 1, L_var_prior = None):
        if L_var_prior is None:
            L_var_prior = np.identity((self.K+self.Qs)*self.lags) # 
        
        if self.normGDP:
            self.GDPmean = np.nanmean(self.GDP)
            self.GDPstd = np.nanstd(self.GDP)
            self.XY[0:,0] = (self.XY[0:,0] - self.GDPmean)/self.GDPstd
        
        XYcomp = self.XY[0:self.GDP.shape[0]*3,:].copy()
        self.XYcomp = _RemoveOutliers(XYcomp, 4, 0)
        self.s0 = s0 # prior for var on loadings
        self.alpha0 = alpha0 # scale on var for loadings
        self.L_var_prior = L_var_prior # var on prior on loading betas (0 by default)

        if self.MAterm == 0:
            self.Hdraw, self.Qdraw, self.Fdraw, self.Pdraw, self.Rdraw, self.Sdraw  = Gibbs_loop(self.XYcomp,F, H, Q, R, S, self.lags, self.lagsH, self.K \
                , self.Qs, s0, alpha0, L_var_prior, ints, burn, save, self.normGDP)
            return self.Hdraw, self.Qdraw, self.Fdraw, self.Pdraw, self.Rdraw, self.Sdraw
        elif self.MAterm == 1:
            self.Hdraw, self.Qdraw, self.Fdraw, self.Pdraw, self.Rdraw, self.Sdraw, self.Phidraw  = Gibbs_loopMA(self.XYcomp,F, H, Q, R, S, self.lags, self.lagsH, self.K \
                , self.Qs, s0, alpha0, L_var_prior, ints, burn, save, self.normGDP)

    def Nowcast(self, start, Horz, burn, save):
        Dates = self.Dates  # 时间序列日期（以季度为单位）
        Findex = int(np.where(Dates == start)[0])  # 找到起始时间点的索引
        Quartersback = (Dates.shape[0] - Findex)  # 从起始点到数据末尾的季度数

        XY, *other_vars = initializevals(Dates.iloc[-1], self.GDP, self.Monthly, self.lags, self.lagsH, self.K, self.Qs)

        # 6. 初始化用于存储预测结果的数组
        Fcast_current = np.empty((Quartersback + 1, 3))  # 当前季度的预测 
        Fcast_next = np.empty((Quartersback + 1, 3))  # 下一季度的预测 
        Outturn_current = np.empty((Quartersback + 1, 3))  # 当前季度的实际值 
        Outturn_next = np.empty((Quartersback + 1, 3))  # 下一季度的实际值 
        # DateFcast = np.arange(start, start + Quartersback / 4, 0.25)  # 预测日期序列
       
        # 7. 循环预测每个时间点的结果
        for ii in range(Quartersback + 1):
            if self.time == start:
                F, H, Q, R, S, ints = self.F, self.H, self.Q, self.R, self.S, self.ints
                self.estimateGibbs(F, H, Q, R, S, ints, burn, save)
            else:
                self.XY, self.H, self.R, self.F, self.Q, self.S, self.ints = initializevals(
                                                                        self.time, self.GDP, self.Monthly, self.lags, self.lagsH, self.K, self.Qs)
                F, H, Q, R, S, ints = self.F, self.H, self.Q, self.R, self.S, self.ints
                self.estimateGibbs(F, H, Q, R, S, ints, burn, save)
                
            # 1. 初始化变量
            Dates = self.Dates  # 时间序列日期（以季度为单位）
            Hdraw = self.Hdraw  # H矩阵的抽样结果（观测方程的系数矩阵）
            Qdraw = self.Qdraw  # Q矩阵的抽样结果（状态方程的误差协方差矩阵）
            Fdraw = self.Fdraw  # F矩阵的抽样结果（状态转移矩阵）
            Rdraw = self.Rdraw  # R矩阵的抽样结果（观测误差协方差矩阵）
            Sdraw = self.Sdraw  # S矩阵的抽样结果（初始状态协方差矩阵）
            Qs = self.Qs  # 状态变量的数量
            K = self.K  # 状态变量的维度

            # 2. 计算参数的中位数（使用贝叶斯抽样结果的中位数作为最终模型参数）
            Hfor = np.median(Hdraw[:,:,:], axis=2)  # H矩阵的中位数
            Qfor = np.median(Qdraw[:,:,:], axis=2)  # Q矩阵的中位数
            Ffor = np.median(Fdraw[:,:,:], axis=2)  # F矩阵的中位数
            Rfor = np.median(Rdraw[:,:,:], axis=2)  # R矩阵的中位数
            Sfor = np.median(Sdraw[:,:,:], axis=2)  # S矩阵的中位数

            # 3. 确保观测数据矩阵 XY 包含足够的预测期（Horz）的空值（NaN）
            Total = Dates.shape[0] * 3 + Horz * 3  # 总时间点数（每季度3个月）
            if XY.shape[0] < Total:
                # 如果观测数据的时间长度不够，则在末尾添加 NaN
                NewNans = np.empty((Total - XY.shape[0], XY.shape[1]))
                NewNans[:] = np.nan
                XYf = np.r_[XY, NewNans]  # 扩展数据矩阵
            else:
                XYf = XY  # 如果数据足够长，则直接使用原始数据

            # 4. 如果启用了 MA（Moving Average）选项，则对数据进行准差分处理
            if self.MAterm == 1:
                Phidraw = self.Phidraw  # MA系数矩阵的抽样结果
                Phifor = np.median(Phidraw[:,:,:], axis=2)  # MA系数矩阵的中位数
                XYf = _quasidiff(XYf, Phifor, Qs)  # 对数据进行准差分处理

            # 5. 确定预测的起始点和回溯的季度数
            Findex = int(np.where(Dates == start)[0])  # 找到起始时间点的索引
            Quartersback = (Dates.shape[0] - Findex)  # 从起始点到数据末尾的季度数
            self.Quartersback = Quartersback  # 保存回溯的季度数

            # 7.1 提取当前的初始状态
            PriorS = Sfor[-1, :].reshape(1, -1)  # 当前时刻的状态

            # 7.2 提取当前时间点的观测数据片段
            if -(Quartersback + Horz) * 3 + (ii * 3 + 3) == 0:
                XYsnap = XYf[-(Quartersback + Horz) * 3 + ii * 3:, :] # 3*7
            else: 
                XYsnap = XYf[-(Quartersback + Horz) * 3 + ii * 3:-(Quartersback + Horz) * 3 + (ii * 3 + 3), :]  # 当前时间点的观测数据

            # 7.3 逐月进行预测
            for jj in range(3):
                # 获取当前季度的实际值
                Outturn_current[ii, jj] = XYsnap[2, 0]  # 当前季度最后一个月的实际值，其每一行元素相同，因为假设月度之和为季度数据
                if ii < Quartersback - 1:
                    # 获取下一季度的实际值 # Outturn_current和Outturn_next将被用于计算误差
                    Outturn_next[ii, jj] = XYf[-(Quartersback + Horz) * 3 + (ii * 3 + 5):-(Quartersback + Horz) * 3 + (ii * 3 + 6), 0]

                # 准备输入数据（Feed），用于卡尔曼滤波
                Feed = XYsnap.copy()
                if jj < 2:
                    Feed[-2 + jj:, :] = np.nan  # 隐藏当前月之后的数据
                Feed[2, 0] = np.nan  # 确保 GDP 数据被隐藏

                # 使用卡尔曼滤波器进行预测
                S_f, P_f = _Kfilter(Feed, Ffor, Hfor, Qfor, Rfor, PriorS)  # 卡尔曼滤波器

                Fcast_current[ii, jj] = np.mean(S_f[0:3, 0])  # 当前季度的预测结果

                print(Fcast_current[ii, jj])

                # 准备下一季度的预测输入
                FeedNext = np.r_[Feed, np.nan * np.empty((3, Feed.shape[1]))]  # 扩展数据以预测下一季度
                S_fn, P_fn = _Kfilter(FeedNext, Ffor, Hfor, Qfor, Rfor, PriorS) 
                Fcast_next[ii, jj] = np.mean(S_fn[3:, 0])  # 下一季度的预测结果

            self.time += 0.25

            print(self.time)
                


        # 8. 如果启用了数据标准化选项，则将预测结果还原到原始尺度
        if self.normGDP:
            Fcast_current = Fcast_current * self.GDPstd + self.GDPmean  # 当前季度预测结果还原
            Fcast_next = Fcast_next * self.GDPstd + self.GDPmean  # 下一季度预测结果还原
            Outturn_current = Outturn_current * self.GDPstd + self.GDPmean  # 当前季度实际值还原
            Outturn_next = Outturn_next * self.GDPstd + self.GDPmean  # 下一季度实际值还原

        # 9. 计算预测误差（RMSE）
        RMSE = np.empty((3, 2))  # 初始化 RMSE 数组
        for ii in range(3):
            CurrentErr = np.square(Fcast_current[:, ii] - Outturn_current[:, ii])  # 当前季度误差平方
            NextErr = np.square(Fcast_next[:, ii] - Outturn_next[:, ii])  # 下一季度误差平方
            RMSE[ii, :] = np.c_[np.sqrt(np.nanmean(CurrentErr)), np.sqrt(np.nanmean(NextErr))]  # 计算 RMSE

        # 10. 扩展日期数组
        Datesaug = np.r_[Dates, Dates[-1:] + 0.25]  # 增加预测期的日期

        # 11. 保存结果到类属性
        self.Fcast_current = Fcast_current  # 当前季度预测结果
        self.Fcast_next = Fcast_next  # 下一季度预测结果
        self.Outturn_current = Outturn_current  # 当前季度实际值
        self.Outturn_next = Outturn_next  # 下一季度实际值
        self.RMSE = RMSE  # 预测误差
        self.Datesaug = Datesaug  # 扩展后的日期序列


    def PlotFcast(self, Month):
        titlelist = ['Month 1', 'Month 2', 'Month 3']
        fig1 = plt.figure(figsize=(15,15))
        ax1 = fig1.add_subplot(111)
        ax1.plot(self.Datesaug[-self.Quartersback-1:], self.Fcast_current[:,Month-1],marker='o', color='olive', linewidth=2)
        ax1.plot(self.Datesaug[-self.Quartersback-1:], self.Outturn_current[:,Month-1], marker='o', color='blue', linewidth=2)
        ax1.legend(['Nowcast', 'Actual'])
        ax1.set_title(titlelist[Month-1])
        plt.show()