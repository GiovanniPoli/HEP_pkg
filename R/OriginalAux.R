RaoBlackwellAuxSMC_witcovs_andNA = function(y, delta,
                                            gamma, x, Z,
                                            matFF, mu0,
                                            Sigma0    = Sigma0, 
                                            SigmaEps  = SigmaEps, 
                                            GG        = Gt, 
                                            seed      = 123,
                                            nSim      = nSim){
  
  # Fixed quantities
  set.seed(seed)
  
  p     = length(mu0)
  TT    = length(y)
  GGt   = t(GG)
  
  # empty quantities
  elapsedTime    = double(TT)
  ESS            = double(length = TT)
  smcF           = array(dim = c(nSim,TT,p)) # nparticles x N x p
  
  # Initialization SMC
  xhat_F    = matrix(rep(mu0, n=nSim), ncol=nSim, nrow=p) #pxn particles
  P_F       = Sigma0
  
  startTime = Sys.time()
  
  const       = c(x%*%delta) + c(Z%*%gamma) # scalar (1x1) + vector (1xn)
  
  
  for(tt in 1:TT){
    
    FF      = matrix(matFF[tt,], nrow = 1) #they are never NA
    FFt     = t(FF)
    FFtFF   = FFt%*%FF
    
    if (is.na(y[tt])) {
      
      xhat_P  = GG%*%xhat_F 
      xPart   = xhat_P + t(mvtnorm::rmvnorm(n = nSim, mean = rep(0,p), 
                                   sigma = SigmaEps)) #test if it works with dimension and so on
      
      xhat_F = xPart
      
    } else {
      
      # Kalman Filter (m=1)
      xhat_P  = GG %*% xhat_F                        # media per theta
      P_P     = GG %*% P_F %*% GGt + SigmaEps        # varianza per theta
      yast_P  = as.vector(FF%*%xhat_P)               # verso y. è la mean per y. questo non funziona
      S_P     = as.double(FF%*%P_P%*%FFt+1)          # per y
      
      # Compute the importance weights
      log.w_IS = pnorm((2*y[tt]-1)*(yast_P + const[tt]) / sqrt(S_P), log.p = T) #è la likelihood come è scritta da durante?
      log.w_IS = log.w_IS - max(log.w_IS)
      w_IS     = exp(log.w_IS)
      
      # Compute ESS
      w_IS_normalized = w_IS/sum(w_IS)
      ESS[tt] = 1/sum(w_IS_normalized^2)
      
      # Resample
      order   = sample.int(nSim, nSim, replace=TRUE, prob=w_IS)
      
      yast_P  = yast_P[order]  
      xhat_P  = xhat_F[,order] 
      
      if (y[tt]==1){
        yast_ = truncnorm::rtruncnorm(n = nSim, a = 0, b = Inf,  mean = yast_P + const[tt], sd = sqrt(S_P))
      } else if (y[tt]==0) {
        yast_ = truncnorm::rtruncnorm(n = nSim, a = -Inf, b = 0, mean = yast_P + const[tt], sd = sqrt(S_P))
      }
      
      xhat_F   = xhat_P + P_P %*% FFt %*% matrix(yast_-yast_P - const[tt],nrow=1,ncol=nSim)/S_P #sun additive representation??
      P_F      = P_P-P_P%*%FFtFF%*%P_P/S_P
      
      xPart   = xhat_F + t(mvtnorm::rmvnorm(n = nSim, mean = rep(0, p), sigma = P_F)) #this I don't understand why is not used for next step
      
    }
    
    smcF[,tt,]     = t(xPart) 
  }
  
  Output = list(filtState     = list(values = smcF,
                                     ESS=ESS),
                elapsedTime   = elapsedTime,
                ConstantTerms = const)
  

  return(Output)
}