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

// [[Rcpp::export]]
arma::vec cpp_rdirichletArma1(arma::colvec parameters){
  int p = parameters.n_elem ;
  arma::vec ret(p);
  for(int i; i<p; i++){
    ret(i) = arma::randg(1, arma::distr_param(parameters(i),1.0))(0);
  }
  ret = ret / arma::sum(ret);
  return ret;
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

// [[Rcpp::export]]
int get_Ri_rcpp( arma::vec gamma, arma::cube Omega){
  
  int Ri; 
  int Ti = Omega.n_cols ;
  
  // sample here or outside?
  arma::mat samp = Omega.row(0) ; 
  arma::vec omegas_sum =  arma::sum(samp , 1) ;
  double avg = arma::mean(omegas_sum) ;
  
  int count = 0;
   
  for (int i = 0; i < Ti; ++i) {
   arma::vec vector_to_check = samp.row(i).t() ;
   if ( arma::all(vector_to_check > 0) ) {
       count++;
     }
   }

   // r1i
   if( (gamma(0) > 0.0) &&  (avg > .1) && (count * 1.0 / Ti > 0.01) ){
     // 0 0 0
     Ri = 0;
   }else if( (gamma(0) > 0.0) &&  (avg > .1) && !(count * 1.0 / Ti > 0.01) ){
     // 0 0 1
     Ri = 1;
   }else if( (gamma(0) > 0.0) &&  !(avg > .1) && (count * 1.0 / Ti > 0.01) ){
     // 0 1 0
     Ri = 2;
   }else if( (gamma(0) > 0.0 )&&  !(avg > .1) && !(count * 1.0 / Ti > 0.01)){
     // 0 1 1
     Ri = 3;
   }else if( !(gamma(0) > 0.0) &&  (avg > .1) && (count * 1.0 / Ti > 0.01) ){
     // 1 0 0
     Ri = 4;
   }else if( !(gamma(0) > 0.0) &&  (avg > .1) && !(count * 1.0 / Ti > 0.01)){
     // 1 0 1 
     Ri = 5;
   }else if( !(gamma(0) > 0.0) &&  !(avg > .1) && (count * 1.0 / Ti > 0.01) ){
     // 1 1 0
     Ri = 6;
   }else if( !(gamma(0) > 0.0) &&  !(avg > .1) && !(count * 1.0 / Ti > 0.01)){
     // 1 1 1 
     Ri = 7;
   }
   
  return Ri;
}

// [[Rcpp::export]]
Rcpp::List SUN_gibbs_sampler( Rcpp::List data,
                              Rcpp::List starting_values, 
                              Rcpp::List priors,
                              double proposal_variance_delta,
                              double proposal_variace_gamma_and_omega,
                              int nSim,
                              int sample,
                              int thinning,
                              int burn){
  
  std::string msg = "Creating Constant for the MCMC...";
  Rcpp::Rcout << msg << std::flush;
  
  // MCMC Variables
  
  const int n = data.length() ;
  Rcpp::List STATE = starting_values;
  // MCMC Stores
  
  arma::mat return_rho(sample, n);
  // arma::mat    BetaReturn  = arma::mat(sample, X.n_cols);
  // arma::colvec sigmaReturn = arma::vec(sample);
  //
  // arma::mat     NewMuRetunrn  = arma::mat(sample, xnew.n_elem);
  // arma::colvec  NewMu         = arma::vec(xnew.n_elem);
  
  
  // MCMC Stores
  
  int SampleStored = 0;
  
  // MCMC settings
  
  
  const int MaxIteration = burn + sample * thinning ;
  
  
  // // MCMC
  
  msg = "\rCreating Constant for the MCMC... Done!";
  Rcpp::Rcout << msg << std::endl;
  msg = "Running the chain...\n";
  Rcpp::Rcout << msg << std::endl;
  
  for (int it = 0; it < MaxIteration; ++it) {
    // MCMC CORE 
    
    // STORE
    if ( it >= burn && ( (it - burn) % thinning == 0 ) ) {
      SampleStored = SampleStored + 1;
    }
    
    Rcpp::Rcout << "\r"  << " " << "[" << it << "/" << MaxIteration << "] \n"  << std::flush;
    
    
    
  }
  
  
  
  Rcpp::Rcout << "\r"   << " " << "[" << MaxIteration << "/" << MaxIteration << "]             "  << std::flush;
  
  return Rcpp::List::create(Rcpp::Named("rho")  = return_rho,
                            Rcpp::Named("Test") = data.length());
}

// [[Rcpp::export]]
arma::ivec rcpp_remove_element(arma::ivec vector, int position) {
  
  int old_ln = vector.n_elem ; 
  arma::ivec ret( old_ln - 1);
  
  if(position == 0){
    ret = vector.subvec(1, old_ln - 1);
  }else if(position == (old_ln - 1) ){
    ret = vector.subvec(0, old_ln - 2);
  }else{
    ret.subvec(0, position - 1) = vector.subvec(0, position - 1);
    ret.subvec(position, old_ln - 2) = vector.subvec(position + 1, old_ln - 1);
  }
  return ret;
}

// [[Rcpp::export]]
arma::mat rcpp_remove_row(arma::mat matrix, int row) {

  
  int old_n_rows = matrix.n_rows ;  
  int n_cols     = matrix.n_cols ;
  
  arma::mat ret( old_n_rows - 1, n_cols ) ;
  
  
  if(row == 0){
    ret = matrix.submat(1,0,old_n_rows-1,n_cols-1);
  }else if(row == (old_n_rows - 1) ){
    ret = matrix.submat(0,0,old_n_rows-2,n_cols-1);
  }else{
    ret.submat(0,0,row-1,n_cols-1) = matrix.submat(0,0,row-1,n_cols-1);
    ret.submat(row,0,old_n_rows-2,n_cols-1) = matrix.submat(row+1,0,old_n_rows-1,n_cols-1);
  }
  return ret;
}

// [[Rcpp::export]]
arma::ivec rcpp_arma_table( NumericVector x) {
  std::map<int, int> counts;

  int n = x.size();
  for (int i = 0; i < n; i++) {
    counts[x[i]]++;
  }

  arma::ivec vec(counts.size());
  int index = 0;
  for (const auto& pair : counts) {
    vec(index++) = pair.second;
  }
  return vec;
}

void rcpp_add_elem_1(arma::ivec& v) {
  v.resize(v.n_rows + 1);  
  v.row(v.n_rows - 1) = 1; 
}

void rcpp_add_row(arma::mat& m, const arma::vec row) {
  m.resize(m.n_rows + 1, m.n_cols); // Aumenta la dimensione di una riga
  m.row(m.n_rows - 1) = row.t(); // Aggiunge la riga alla fine
}

// [[Rcpp::export]]
Rcpp::List test_function(arma::mat m, arma::vec row, arma::ivec ivector){
  rcpp_add_row(m, row) ;
  rcpp_add_elem_1(ivector);
  return List::create( Rcpp::Named("1")  = m , 
                       Rcpp::Named("2")  = ivector );
}

// [[Rcpp::export]]
arma::ivec rcpp_correct_labels( arma::ivec x, int target) {
  
  int n = x.n_elem ;

  for (int i = 0; i < n; i++) {
    if( x(i) > target){
      x(i) -= 1 ;
    }
  }
  return x;
}

arma::vec rcpp_update_delta( Rcpp::List state,
                             Rcpp::List data,
                             Rcpp::List prior,
                             double sd_rw){
  
  Rcpp::List ALL_subject_par = state["subjects"] ;
  Rcpp::CharacterVector IDs  = ALL_subject_par.names() ;
  int n_subject = IDs.size()   ; 
  Rcpp::List  SubjectItems ;
  std::string key ;

  double old_ll = 0.0 ;
  double new_ll = 0.0 ;
  
  arma::colvec d      = state["delta"] ;
  int p = d.n_elem ;
  
  arma::colvec new_d  = d + arma::randn(p) * sd_rw ;
  arma::colvec vec_dif = new_d - d ;
  arma::colvec next_delta   ;
  arma::vec old_lin_pred ;
  arma::vec new_lin_pred ;
  
  arma::rowvec xi ;
  arma::vec y ;
  arma::vec probs ;
  double shift_lin_pred ;
  
  
  for (int subject = 0; subject < n_subject; ++subject){
    
    key = as<std::string>(IDs[subject]) ;
    SubjectItems = ALL_subject_par[ key ] ;
  
    old_lin_pred = Rcpp::as<arma::vec>(SubjectItems["eta"]) ;

    //xi = data["x"]  ;
    //y  = data["ts"] ; 
//
  //  new_lin_pred = old_lin_pred + xi * vec_dif ;
  //  probs        = arma::normcdf(new_lin_pred) ;
  //  
  //  old_ll += SubjectItems["log.lik"] ;
  //  new_ll += y % arma::log(probs) + (1-y) % arma::log(1-probs);
  //  
  }
    
  return next_delta;
}

// [[Rcpp::export]]
Rcpp::List rcpp_update_rho_and_xi( Rcpp::List state, 
                                   arma::ivec gTable,
                                   Rcpp::List prior) {
  arma::vec a = prior["a"];
  double M = prior["M"];
  double sigma = prior["sigma"];
  
  arma::ivec rho = state["rho"] ;
  arma::mat  xi  = state["Xi"]  ;
  Rcpp::List ALL_subject_par = state["subjects"] ;
  Rcpp::CharacterVector IDs = ALL_subject_par.names() ;
  
  
  
  int H = xi.n_rows ;
  int n_subject = rho.n_elem   ; 
  
  int rho_s ;
  int new_label;
  int Ri_col ;
  
  
  arma::vec pnew =  a / arma::sum( a );
  arma::colvec new_xi_star ;
  Rcpp::List SubjectItems ;
  std::string key ;
  arma::colvec probs ;
  arma::uvec labels ;
  
  
  for (int subject = 0; subject < n_subject; ++subject){

    rho_s = rho(subject) ;

    key = as<std::string>(IDs[subject]) ;
    SubjectItems = ALL_subject_par[ key ] ;
    Ri_col = SubjectItems["Ri"] ;
    
    if( gTable(rho_s) == 1){

       gTable = rcpp_remove_element( gTable, rho_s) ;
       xi     = rcpp_remove_row( xi, rho_s) ;

       rho    = rcpp_correct_labels( rho, rho_s) ;
       H      -= 1 ;
       
    }else{
       gTable(rho_s) -= 1 ;
    }   
    
    probs.resize(H + 1);
    probs.subvec(0,H-1)  = ( arma::conv_to<arma::vec>::from(gTable) - sigma) %
                             xi.col(Ri_col) ; 
    probs(H) = pnew(Ri_col) * ( M + H * sigma);
    
    labels    = rcpp_index_gen( H+1 ) ;
    new_label = rcpp_sample(labels,1, probs)(0);
    
    if(new_label == H){
      
      rcpp_add_elem_1(gTable);
      
      new_xi_star = cpp_rdirichletArma1(a) ;
      rcpp_add_row(xi, new_xi_star) ;
      H += 1 ;
    
    }else{
      gTable(new_label) += 1 ;
    }
    
    rho(subject) = new_label;
  }
  

  return List::create( Rcpp::Named("table") = gTable, 
                       Rcpp::Named("Xi")    = xi,
                       Rcpp::Named("rho")   = rho);
}
