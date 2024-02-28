#include <iostream>
#include <chrono>
#include <thread>

#include <RcppArmadillo.h>
#include <truncnorm.h>
#include <mvnorm.h>
#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]


arma::mat cpp_mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}
arma::mat cpp_mvrnormArma_transposed(int n, arma::vec mu, arma::mat sigma) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(ncols, n);
  return arma::repmat(mu, 1, n) + arma::chol(sigma).t() * Y ;
}

//' @name mvrnormArma
//' @title Samples from mutivariate normal using Rcpp and Armadillo
//' @description Function to return 1 or n samples from a mutivariate normal using Rcpp and Armadillo
//' @param n number of samples .
//' @param mu Mean vector.
//' @param sigma Covariance matrix vector.
//' @return Random samples
//' @examples mvrnormArma(n = 1, mu = c(0,0), sigma = diag(c(1,1)))
//' @field mvrnormArma Return n sample from a mutivariate normal.
//' @field mvrnormArma1 Return a sample from a mutivariate normal.
// [[Rcpp::export]]
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
  return cpp_mvrnormArma(n, mu, sigma);
}
arma::colvec cpp_mvrnormArma1(arma::vec mu, arma::mat sigma) {
  int ncols = sigma.n_cols;
  arma::colvec Z = arma::randn(ncols);
  return mu + arma::chol(sigma).t() * Z;
}


//' @rdname mvrnormArma
// [[Rcpp::export]]
arma::colvec mvrnormArma1(arma::vec mu, arma::mat sigma) {
  return cpp_mvrnormArma1(mu, sigma);
}

arma::uvec rcpp_index_gen(const int n) {
  arma::uvec ret(n);
  for(int i = 0; i < n; ++i) {
    ret(i) = i ;    
  }
  return(ret);
}

// [[Rcpp::export]]
arma::uvec rcpp_sample( const arma::uvec vec,
                        const int size,
                        const arma::colvec prob) {
  arma::uvec ret = RcppArmadillo::sample(vec, size, true, prob);
  return(ret);
}

//' @name conditioned_C_RaoBlackwellAuxSMC_witcovs_andNA
//' @title Sample nSim MC samples from a SUN with time constant terms using Rcpp and Armadillo.
//' @description Function Sample nSim MC samples from a SUN with time constant terms using Rcpp and Armadillo.
//' @param y Binary vector of 1,0 or NA 
//' @param delta Vector for time constant covariates.
//' @param gamma Vector for time varying covariates. 
//' @param x Vector encoding time constant covariates.
//' @param Z Matrix encoding time varing covariates, columns are observed times.
//' @param matFF Encoding of evolutions coefficents.
//' @param G Evolution matrix.
//' @param mu0 Prior mean for starting time.
//' @param Sigma0 Prior variance for starting time.
//' @param SigmaEps Evolution prior.
//' @param nSim Number of samples.
//' @param seed seed to be fixed.
//' @return Random samples of evoution coefficents.
//' @examples 1+1 == 2
//' @field conditioned_C_RaoBlackwellAuxSMC_witcovs_andNA Rcpp Function.
//' @field RaoBlackwellAuxSMC_witcovs_andNA Original R function.

// [[Rcpp::export]]
arma::cube conditioned_C_RaoBlackwellAuxSMC_witcovs_andNA( const Rcpp::IntegerVector y,
                                                           const arma::colvec delta,
                                                           const arma::colvec gamma,
                                                           const arma::colvec x,
                                                           const arma::mat Z,
                                                           const arma::mat matFF,
                                                           const arma::mat G,
                                                           const arma::colvec mu0,
                                                           const arma::mat Sigma0, 
                                                           const arma::mat SigmaEps,
                                                           const int nSim = 100,
                                                           const int seed = 1) {
  
  // Const.
  const int p  = mu0.n_elem  ;
  const int TT = y.length()  ;
  const arma::uvec x_order   = rcpp_index_gen(nSim) ;
  
  const arma::mat Gt = G.t() ;
  const arma::colvec LinerPred_noEv = arma::as_scalar( x.t() * delta ) + Z * gamma ;
  const arma::colvec zero_ex_ev = arma::zeros<arma::vec>(p);
  
  // Init. the MC sample 
  arma::cube SMCret( nSim, TT, p );
  
  // SMC stuffs
  arma::mat  xhat_F = arma::repmat(mu0, 1, nSim);
  arma::mat  P_F    = Sigma0;
  
  arma::colvec FF(p);
  arma::rowvec FFt(p);
  arma::mat    FFtFF(p, p);
  
  arma::mat xhat_P(p, nSim);
  arma::mat P_P(p, p);
  arma::colvec yast_P(nSim);
  arma::colvec log_ws_IS_it(nSim);
  arma::colvec yast_(nSim);
  double S_P;
  
  arma::colvec xhat_col_vec(nSim);
  arma::mat xPart(p, nSim);
  
  arma::uvec order_rss(nSim);
  
  for(int tt = 0; tt < TT; ++tt) {
    
    FFt   = matFF.submat(tt, 0, tt, p-1) ;
    FF    = FFt.t();
    FFtFF = FF * FFt;
    
    if( y(tt) == NA_INTEGER){
      
      // Prior Evaluate evolutions
      
      xhat_P  = G * xhat_F; 
      xPart   = xhat_P + cpp_mvrnormArma_transposed(nSim, zero_ex_ev, SigmaEps) ;
      xhat_F  = xPart ;
      
    }else if( y(tt) == 1 ){
      
      // Evaluate Evolutions
      
      xhat_P  = G * xhat_F;                      
      P_P     = G * P_F * Gt + SigmaEps;   
      yast_P  = (FFt * xhat_P).t();
      S_P     = arma::as_scalar(FFt*P_P*FF + 1.0);
      
      
      // Compute weights
      log_ws_IS_it = arma::log(arma::normcdf( 
        (2.0 * y(tt) - 1.0) * (yast_P + LinerPred_noEv(tt) )
                                 / std::sqrt(S_P) )) ; 
      log_ws_IS_it = log_ws_IS_it - arma::max(log_ws_IS_it);
      
      // Resample step (y_{it} = 1)
      order_rss = rcpp_sample(x_order, nSim, arma::exp(log_ws_IS_it) );
      yast_P    = yast_P(order_rss);
      xhat_P    = xhat_F.cols(order_rss);
      
      // Update from [0, Inf] truncated normal
      for(int sample_to_evol = 0; sample_to_evol < nSim; ++sample_to_evol) {
        yast_(sample_to_evol) = r_truncnorm( yast_P(sample_to_evol) + LinerPred_noEv(tt), 
              std::sqrt(S_P),
              0, R_PosInf);
      }
      
      // MC Evolution
      xhat_col_vec = yast_ - yast_P - LinerPred_noEv(tt)   ;
      xhat_F       = xhat_P + P_P * FF * xhat_col_vec.t() / S_P ;
      P_F          = P_P - P_P * FFtFF * P_P / S_P;
      xPart        = xhat_F + cpp_mvrnormArma_transposed(nSim, zero_ex_ev, P_F) ;
      
    }else{
      
      // Evaluate Evolutions
      xhat_P  = G * xhat_F;                      
      P_P     = G * P_F * Gt + SigmaEps;    
      yast_P  = (FFt * xhat_P).t();
      S_P     = arma::as_scalar(FFt*P_P*FF + 1.0);
      
      // Compute weights
      log_ws_IS_it = arma::log(arma::normcdf( 
        (2.0 * y(tt) - 1.0) * (yast_P + LinerPred_noEv(tt) )
                                 / std::sqrt(S_P) )) ; 
      log_ws_IS_it = log_ws_IS_it - arma::max(log_ws_IS_it);
      
      // Resemple step (y_{it} = 0)
      order_rss = rcpp_sample(x_order, nSim, arma::exp(log_ws_IS_it));
      
      yast_P    = yast_P(order_rss);
      xhat_P    = xhat_F.cols(order_rss);
      // Update from [-Inf, 0] truncated normal
      for(int sample_to_evol = 0; sample_to_evol < nSim; ++sample_to_evol) {
        yast_(sample_to_evol) = r_truncnorm( yast_P(sample_to_evol) + LinerPred_noEv(tt), 
              std::sqrt(S_P),
              R_NegInf, 0);
      }
      
      
      // MC Evolution
      xhat_col_vec = yast_ - yast_P - LinerPred_noEv(tt)   ;
      xhat_F       = xhat_P + P_P * FF * xhat_col_vec.t() / S_P ;
      P_F          = P_P - P_P * FFtFF * P_P / S_P;
      xPart   = xhat_F + cpp_mvrnormArma_transposed(nSim, zero_ex_ev, P_F) ;
      
    }
    SMCret.col(tt) = xPart.t() ;
  }
  return SMCret;
}

