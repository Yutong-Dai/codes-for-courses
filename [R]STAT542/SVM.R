#!/usr/bin/env R
# -*- coding: utf-8 -*-
# @Date    : 2017-11-05 22:47:01
# @Author  : Roth (rothdyt@gmail.com)
# @Version : 0.9

MySvm <- function(x,y,c){
  # linear kernel svm
  # @ parameters:
  # x: design matrix of size n*p
  # y: label
  # c: penalty strength; the large it is, the more penalty you put on slack variables
  
  # @ value
  # sol.dual: dual problem's solution
  # coef: coefficients of the seperating hyper-plane
  # index: support vectors' indice
  require(quadprog)
  n <- nrow(x); p <- ncol(x)
  matrix_list <- list()
  for (i in seq_len(nrow(x))){
    matrix_list[[i]] <-  y[i]*x[i,]
  }
  Diagmat <- as.matrix(Matrix::bdiag(matrix_list))
  Idenmat <- diag(1,p); Idenmat <- do.call("rbind", rep(list(Idenmat), n))
  Idenmat <- do.call("cbind", rep(list(Idenmat), n))
  Dmat <- t(Diagmat) %*% Idenmat %*% Diagmat + diag(1e-5,n)
  dvec <- rep(1,n)
  Amat <- rbind(t(y), diag(1,n)); Amat <- t(rbind(Amat,diag(-1,n)))
  bvec <- c(rep(0,n+1),rep(-c,n))
  sol.dual <- solve.QP(Dmat, dvec, Amat, bvec, meq=1, factorized=FALSE)$solution
  sol.dual.beta <- t(x) %*% (y * sol.dual)
  label <- (sol.dual > 1e-5 & sol.dual < c - 1e-5)
  sol.dual.beta_0 <- mean(1 / y[label] - x[label,] %*% sol.dual.beta)
  results <- list(sol.dual=sol.dual,
                  coef=c(sol.dual.beta_0,sol.dual.beta),
                  index=label)
  class(results) <- "mysvm"
  return(results)
}

MySvmPredic <- function(mysvmfit,x.pred,y.pred=NULL){
  # using fitted svm model to make prediction
  # @ parameters:
  # x.pred: design matrix of size n*p; you want to classify these points
  # y.pred: label
  
  # @ value
  # predict: predicted categories
  # accuracy: if y.pred is provided, then the prediciton accuracy is provided
  fitted <- cbind(1,x.pred) %*% mysvmfit$coef
  category <- rep(-1,nrow(x.pred))
  category[fitted>0] <- 1
  if (!is.null(y.pred)){
    accuracy <- sum(category == y.pred) / length(y.pred)
    confusion.matrix <- table(y.pred,category,dnn=c("Actual", "Predicted"))
    results <- list(predict=category,accuracy=accuracy)
  }else{
    results <- list(predict=category) 
  }
  return(results)
}

svm.cv <- function(x,y,crange=10^(-6:6),K=10){
  # using k-fold cv to select parameter c
  # @ parameters:
  # x: design matrix of size n*p
  # y: label vectors of size n*1; only takes value at {-1, +1}
  # crange: range of c's you consider; a vector of size t*1
  # K: k-fold cv; specify k
  
  # @ value
  # best.c: best c gives your highest prediction accuracy
  # best.accuracy: best prediction accuracy

  # split the data set
  if (!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
  runif(1)
  seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  n <- nrow(x)
  if ((K > n) || (K <= 1))
  stop("'K' outside allowable range")
  K.o <- K
  K <- round(K)
  kvals <- unique(round(n/(1L:floor(n/2))))
  temp <- abs(kvals - K)
  if (!any(temp == 0))
    K <- kvals[temp == min(temp)][1L]
  if (K != K.o)
    warning(gettextf("'K' has been set to %f", K), domain = NA)
  f <- ceiling(n/K)
  s <- sample(rep(1L:K, f), n)
  ms <- max(s)
  t <- length(crange)
  accuracy.vec <- rep(0,t)
  count <- 0
  for (c in crange){
    accuracy <- 0
    for (i in seq_len(ms)) {
      j.out <- seq_len(n)[(s == i)]
      j.in <- seq_len(n)[(s != i)]
      x.i<-x[j.in,,drop=FALSE]
      y.i<-y[j.in,,drop=FALSE]
      svmfit.i <- MySvm(x.i,y.i,c)
      pred.i <- MySvmPredic(svmfit.i,x.pred=x[j.out,,drop=FALSE],
                            y.pred=y[j.out,,drop=FALSE])
      accuracy <- accuracy + pred.i$accuracy 
    }
    count <- 1 + count
    accuracy.vec[count] <- accuracy / K
  }
  best.c <- crange[which.max(accuracy.vec)]
  results <- list(best.accuracy=max(accuracy.vec), best.c = best.c, 
    accuracy.vec = accuracy.vec)
  return(results)
}