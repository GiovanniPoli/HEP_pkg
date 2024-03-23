library(testthat)

settings = list()
settings$delta = c(-1.281552,0)
settings$gamma =  -1.044796
settings$Z = t(matrix(c(rep(c(0), 365),
                              rep(c(1), 365)), 
                            nrow  = 1, ncol = 365*2, 
                            byrow = F, dimnames = NULL))
settings$matFF = matrix(1, nrow = 730, ncol = 10)
settings$mu0       = rep(0,10)
settings$Sigma0    = diag(rep(1,10)) / 1000 
settings$SigmaEps  = diag(rep(1,10)) / 1000
settings$x  = c(1,1)
G = matrix(0, ncol = 10, nrow = 10) ; G[2:10,1:9] = diag(rep(1,9)) 
settings$G = G
y = c(rbinom(365,1,  .1), rbinom(365,1, .01)) ; y[180:200] = 1
settings$y = y
remove(y,G)

set.seed(1)

A = HEP::conditioned_C_RaoBlackwellAuxSMC_witcovs_andNA(y    = settings$y,
                                                   delta     = settings$delta,
                                                   gamma     = settings$gamma,
                                                   x         = settings$x ,
                                                   Z         = settings$Z,
                                                   matFF     = settings$matFF,
                                                   mu0       = settings$mu0,
                                                   Sigma0    = settings$Sigma0, 
                                                   SigmaEps  = settings$SigmaEps, 
                                                   G         = settings$G, 
                                                   seed = 1,
                                                   nSim = 100)
  

set.seed(1)

B = HEP::conditioned_C_RaoBlackwellAuxSMC_witcovs_andNA(y    = settings$y,
                                                        delta     = settings$delta,
                                                        gamma     = settings$gamma,
                                                        x         = settings$x ,
                                                        Z         = settings$Z,
                                                        matFF     = settings$matFF,
                                                        mu0       = settings$mu0,
                                                        Sigma0    = settings$Sigma0, 
                                                        SigmaEps  = settings$SigmaEps, 
                                                        G         = settings$G, 
                                                        seed = 1,
                                                        nSim = 100)




test_that("Method return numerical values:", {expect_type( A ,"double")})
test_that("Rcpp Seed:", {expect_true(identical(A,B))} )
