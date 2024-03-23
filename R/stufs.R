
data_list = list("A1" = list(y   = rbinom(730, 1, 0.01),
                             x   = c(1,1),
                             Z   = rep(c(0,1), each = 365),
                             gli = c(rep(0.1,7),.3)) ,
                 "A2" = list(y   = rbinom(730, 1, 0.1),
                             x   = c(1,1),
                             Z   = rep(c(0,1), each = 365),
                            gli = c(rep(0.1,7),.3)))
              
PRIORS = list()

set.seed(1)
starting_values = list("subjects" = lapply(1:30, function(x){
                                        list(gamma       = 1,
                                             mega_index = 1,
                                             Ri = 2,
                                             PARTICLES   = NULL)}),
       "delta"    = c(0,1),
       "Xi"       = matrix(c(rep(0,8)), nrow = 3, ncol = 8, byrow = T),
       "rho"      = sample(rep(c(0,1,2),each = 10)))
starting_values$Xi[1,1] = 1
starting_values$Xi[2,2] = 1
starting_values$Xi[3,3] = 1

names(starting_values$subjects) = names(starting_values$rho) =  paste0("A.",1:30)


starting_values$subjects$A.1$Ri
starting_values$rho
starting_values$Xi
table(starting_values$rho)

if(FALSE){
  SUN_gibbs_sampler( data = data_list,
                     starting_values = starting_values, 
                     priors = PRIORS,
                     proposal_variance_delta = 1,
                     proposal_variace_gamma_and_omega = 1,
                     nSim = 100,
                     sample = 1000,
                     thinning = 1,
                     burn = 1 ) 
  
  A = rcpp_update_rho_and_xi(state = starting_values,
                          prior =  list("a" = rep(1,8),
                                        "M" = 5,
                                        "sigma" = 0),
                         gTable =  table(starting_values$rho)
  )
  
  t0 = Sys.time()
  rowMeans(apply(replicate(1000000, cpp_rdirichletArma1(c(1,1,2))),3,function(x) x))
  t1 = Sys.time()
  t1-t0
  t2 = Sys.time()
  rowMeans(apply(replicate(1000000, cpp_rdirichletArma1_wrong(c(1,1,2))),3,function(x) x))
  t3 = Sys.time()
  t3-t2
}


