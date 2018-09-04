enet <- function (x,y,alpha=1,lambda=NULL,nlambda=NULL){
  # ------------------------- function document --------------------------------
  # using CD algorithm to find solution path for linear regression problems
  # with mixtures of l1 and l2 penalty
  #
  # @Args      :
  #     x     :  Design matrix of size n*p
  #     y     :  Observation matrix of size n*1
  # alpha     :  A parameter in the [0,1] to balance the strength of 
  #              l1 penalty and l2 penalty.
  #              alpha = 1, lass0; alpha = 0, ridge.
  # lambda    :  A parameter to control the strength of overall penalty
  #              The lager lambda is, the stronger the penalty is.
  #              If you provide this value, a specific lambda will be used.
  # nlambda   :  If this parameter is provided, then a sequence of lambda
  #              will be automatically computed and used to fit a series 
  #              of model.

  # Outputs   :
  # for single lambda case:  
  #   time    : running time
  # hatbeta   : the estimated parameter
  # iteration : the total iteration in this run
  # for a sequence of lambda case:
  # beta_all   : a matrix of fitted beta, including the intercept; 
  #              all are returned to the orginal scale.
  #              each column corresponding to a fitted beta for a model
  # lambda_seq: the lambda sequence used to compute the elastic net;
  #             all are returned to the original scale. 
  # nonzero:    number of parameters that are nonzero.
  
  # ------------------------- program information ------------------------------
  # author: Yutong Dai <yutongd3@illinois.edu>
  # version: 1.0
  # last update: 2017.10.2
  # ------------------------- input inspection ---------------------------------
  if (class(x) != "matrix" | class(y) != "matrix"){
    stop("both input x and input y should be in the matrix form")
  }
  # -------------------------------- sub-functions -----------------------------
  softT<-function(z,r){
    if(z > 0 && r < abs(z) ){
      z-r
    }else{
      if(z < 0 && r < abs(z) ){
        z + r
      }else{
        0
      }
    }
  }

  # data preprocess
  n <- nrow(x); p <- ncol(x)
  mean.y <- mean(y)
  mean.x <- apply(x, 2, mean)
  # scale the data in accordance with the thesis
  sigmax <- apply(x, 2, sd) * sqrt((n - 1) / n) 
  sigmay <- sd(y) * sqrt((n-1) / n) 
  x.old <- x; y.old <- y
  x <- scale(x) * sqrt(n) / sqrt(n - 1) 
  y <- y / sigmay
  # ------------------------------ Algorithms ----------------------------------
  if (!is.null(lambda)){
    # parameter initialization
    lambda <- lambda / sigmay
    # opeartion indicated by the <http://stats.stackexchange.com/questions/155362/glmnet-unstandardizing-linear-regression-coefficients>
    beta_present <- rep(0,p); beta_update <- rep(0,p)
    op <- 1; it <- 0
    # count time
    timestart <- Sys.time(); 
    # covariance update begins here
    crossprod.xy <- t(x) %*% y
    inner.x <- t(x) %*% x
    while(op == 1){
      for (j in seq(1:p)){
        z <- (crossprod.xy[j] - inner.x[j,] %*% beta_update) / n + beta_update[j]
        beta_update[j] <- (softT(z,lambda * alpha)) / (1 + lambda * (1 - alpha))
      }
      if (max(abs(beta_update - beta_present))>1e-6){
        beta_present <- beta_update
      }else{
        op <- 2
      }
      it <- it + 1
    } 
    beta_rescale <- beta_update * sigmay / sigmax
    intercept <- mean(y.old) - colMeans(x.old) %*% beta_rescale
    timeend <- Sys.time()
    runningtime <- timeend - timestart
    stata <- list()
    stata$time <- runningtime
    stata$hatbeta <- as.matrix(c(intercept,beta_rescale))
    stata$iteration <- it
    return(stata) 
  }else{
    # using a sequence of lambda
    lammax <- max(abs(t(x)%*%y))/n; lammin <- 0.001 * lammax
    lambda <- exp(seq(log(lammax), log(lammin), length=nlambda))
    beta_sequence <- matrix(NA,ncol=nlambda,nrow=p)
    intercept_sequence <- matrix(NA,ncol=nlambda,nrow=1)
    nonzero <- rep(0,nlambda)
    crossprod.xy <- t(x) %*% y
    inner.x <- t(x) %*% x
    for (i in seq_len(nlambda)){
      if (i == 1){
        beta_present <- rep(0,p)
        beta_update <- rep(0,p)
      }
      beta_present <- beta_update
      op <- 1
      while(op == 1){
        for (j in seq(1:p)){
          z <- (crossprod.xy[j] - inner.x[j,] %*% beta_update) / n + beta_update[j]
          beta_update[j] <- (softT(z,lambda[i] * alpha)) / (1 + lambda[i] * (1 - alpha))
        }
        if (max(abs(beta_update - beta_present))>1e-6){
          beta_present <- beta_update
        }else{
          op <- 2
        }
      }
      beta_sequence[,i] <- beta_update * sigmay / sigmax
      intercept_sequence[,i] <- mean(y.old) - colMeans(x.old) %*% beta_sequence[,i]
      nonzero[i] <- p - sum(abs(beta_sequence[,i]) <= 1e-6) 
    }
    beta_all <- rbind(intercept_sequence,beta_sequence)
    xnames <- vector(length=p)
    for (i in seq_len(ncol(x))){xnames[i]<-paste("x",i,sep="")}
    rownames(beta_all) <- c("intercept",xnames)
    stata <- list(beta_all=beta_all,lambda_seq=lambda*sigmay,nonzero=nonzero)
    return(stata)
  }
}