#!/usr/bin/env R
# -*- coding: utf-8 -*-
# @Date    : 2017-11-05 22:38:16
# @Author  : Roth (rothdyt@gmail.com)
# @Version : 0.9

KernelRegression <- function(x,y,x0,bandwidth){
  # ------------------------- function document --------------------------------
  # Perform univariate Gaussian Kernel Regression
  #
  # @Args:
  # x:  predictor; matrix n*1
  # y:  response; matrix n*1
  # x0: points you want to predict; matix m*1	
  # bandwidth: bandwidth used in the kernel function
  
  # @values:
  #	fval: prediced value at points x0
  x0 <- matrix(x0,ncol=1)
  x <- matrix(x,ncol=1)
  y <- matrix(y,ncol=1)
  GaussianKernel <- function(xi,x0,bandwidth){
    bandwidth <- 0.3706506 * bandwidth
    distance <- (xi - x0) / bandwidth
    kval <- 1 / sqrt(2 * pi) * exp(- distance^2/2)
    # prevent zero 
    label <- which(kval <= 1e-12)
    kval[label] <- 1e-12
    return(kval / bandwidth)
  }
  fval <- rep(0,length(x0))
  for (j in seq_len(length(x0))){
    weights <- sapply(x,GaussianKernel,x0[j],bandwidth)
    nomralizer <- sum(weights)
    weights <- weights / nomralizer
    fval[j] <- t(weights) %*% y
  }
  return(fval)
}
KR.cv <- function(x,y,bandwidthrange,K=10){
  x <- matrix(x,ncol=1)
  y <- matrix(y,ncol=1)
  if (!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
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
  MSE <- rep(0,length(bandwidthrange))
  sd <- rep(0,length(bandwidthrange))
  for (t in seq_len(length(bandwidthrange))){
    mse.t <- matrix(NA,nrow = K,ncol=1)
    for (i in seq_len(ms)){
      j.out <- seq_len(n)[(s == i)]
      j.in <- seq_len(n)[(s != i)]
      x.i<-x[j.in,,drop=FALSE]
      y.i<-y[j.in,,drop=FALSE]
      y.pred.i <- KernelRegression(x.i,y.i,x[j.out,,drop=FALSE],
                                   bandwidth=bandwidthrange[t])
      y.test.i <- y[j.out,,drop=FALSE]
      mse.t[i,1] <- sum((y.pred.i - y.test.i)^2)/length(y.test.i) 
    }
    MSE[t] <- sum(mse.t) / K
    sd[t] <- sd(mse.t)
  }
  index <- which.min(MSE)
  best.bandwidth <- bandwidthrange[index]
  results <- list(best.bandwidth=best.bandwidth,
                  bandwidthrange=bandwidthrange,
                  MSE=MSE,sd=sd)
  class(results) <- "KernelRegression"
  return(results)	
}

plot.kg <- function(KernelRegressionObject,onesd=TRUE){
  xstart <- KernelRegressionObject$bandwidthrange[which.min(
    KernelRegressionObject$MSE)] * 0.95
  xend <- KernelRegressionObject$bandwidthrange[which.min(
    KernelRegressionObject$MSE)] * 1.05
  yupper <- min(KernelRegressionObject$MSE)+
    KernelRegressionObject$sd[which.min(
      KernelRegressionObject$MSE)]
  ylower <- min(KernelRegressionObject$MSE)-
    KernelRegressionObject$sd[which.min(
      KernelRegressionObject$MSE)]
  plot(KernelRegressionObject$bandwidthrange,
       KernelRegressionObject$MSE,type="l",
       ylim=c(ylower*0.9,max(KernelRegressionObject$MSE)*1.1),
       xlab="band width",ylab="MSE")
  segments(x0=xstart,y0=yupper,x1=xend)
  segments(x0=xstart,y0=ylower,x1=xend)
  segments(x0=KernelRegressionObject$bandwidthrange[which.min(
    KernelRegressionObject$MSE)],y0=ylower,y1=yupper)
  if (onesd){
    label <- which(KernelRegressionObject$MSE < yupper & 
                     ylower < KernelRegressionObject$MSE)
    if (length(label) != 0){
      label <- max(label)
      bandwith.suggest <- KernelRegressionObject$bandwidthrange[label]
      abline(v=bandwith.suggest,col="red")
    }else{
      bandwith.suggest <- KernelRegressionObject$bandwidthrange[which.min(
        KernelRegressionObject$MSE)]
      abline(v=bandwith.suggest,col="red")
    }
    print(paste("Suggested band width is",round(bandwith.suggest,digits=2)))
  }
}

KR.cv.fast <- function(x,y,bandwidthrange,K=5){
  y <- y[order(x)]
  x <- x[order(x)]
  x <- matrix(x,ncol=1)
  y <- matrix(y,ncol=1)
  if (!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
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
  MSE <- rep(0,length(bandwidthrange))
  sd <- rep(0,length(bandwidthrange))
  for (t in seq_len(length(bandwidthrange))){
    mse.t <- matrix(NA,nrow = K,ncol=1)
    for (i in seq_len(ms)){#browser()
      j.out <- seq_len(n)[(s == i)]
      j.in <- seq_len(n)[(s != i)]
      x.i<-x[j.in,,drop=FALSE]
      y.i<-y[j.in,,drop=FALSE]
      y.pred.i <- ksmooth(x.i, y.i, kernel = "normal", 
                          bandwidth = bandwidthrange[t], x.points = x[j.out,,drop=FALSE])$y
      y.test.i <- y[j.out,,drop=FALSE]
      label.i <- which(is.na(y.pred.i))
      if (length(label.i)>0){
        if (length(label.i)==length(y.pred.i)){
          y.pred.i <- rep(mean(y.i),length(label.i))
        }else{
          y.pred.i <- y.pred.i[-label.i]
          y.test.i <- y.test.i[-label.i]
        }
      }
      mse.t[i,1] <- sum((y.pred.i - y.test.i)^2)/length(y.test.i) 
    }
    
    MSE[t] <- sum(mse.t) / K
    sd[t] <- sd(mse.t)
  }
  index <- which.min(MSE)
  best.bandwidth <- bandwidthrange[index]
  results <- list(best.bandwidth=best.bandwidth,
                  bandwidthrange=bandwidthrange,
                  MSE=MSE,sd=sd)
  class(results) <- "KernelRegression"
  return(results)	
}


# test
set.seed(1234)
x <- runif(100, 0, 2*pi)
x <- x[order(x)]
y <- 2 * sin(x) + rnorm(length(x))
bandwidthrange = seq(0.001,20,length.out = 50)
set.seed(1)
r <- KR.cv(x,y,bandwidthrange,K=10)
plot.kg(r)



