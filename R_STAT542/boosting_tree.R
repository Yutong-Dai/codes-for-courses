#!/usr/bin/env R
# -*- coding: utf-8 -*-
# @Date    : 2017-11-14 20:40:42
# @Author  : Roth (rothdyt@gmail.com)
# @Version : 0.9

stump <- function(x,y,w,xpred=NULL){
  # A stump model for classification with just one split
  # for one dimensional problem.
  
  # @parameters:
  # x: feature vector; n*1
  # y: label vector; n*1
  # w: weights vector; n*1
  # xpred: points to be predicted  
  
  # @values:
  # cut_point: cutting poings; 1*1
  # fval_left: left node predictions; character
  # fval_right: right node predictions; character
  Gini <- function(y0,w0){
    # calculate weighted Gini index for curret node
    # @values:
    # return the Gini Index for current node
    p <- (t(w0) %*% (y0==1)) / (sum(w0))
    return(p * (1 - p))
  }
  y <- y[order(x)]
  w <- w[order(x)]
  x <- x[order(x)]
  xrange <-unique(x); xrange <- xrange[-length(xrange)]
  left <- rep(NA,length(xrange)); right <- left
  score_mat <- matrix(NA,nrow=length(xrange),ncol=2)
  count <- 0
  for (c in xrange){#browser()
    split_idx <- max(which(x==c))
    y_left <- y[1:split_idx]; y_right <- y[-c(1:split_idx)]
    w_left <- w[1:split_idx]; w_right <- w[-c(1:split_idx)]
    Gini_left <- Gini(y_left, w_left); Gini_right <- Gini(y_right, w_right)
    count <- count + 1
    score_mat[count,1] <- sum(w_left) / sum(w) * Gini_left + 
      sum(w_right) / sum(w) * Gini_right
    score_mat[count,2] <- c # cut point
    p.left <- sum(w_left[y_left==1])/sum(w_left)
    p.right <- sum(w_right[y_right==1])/sum(w_right)
    left[count] <- ifelse(p.left >= 0.5, 1, -1)
    right[count] <- ifelse(p.right >= 0.5, 1, -1)
  }
  idx <- which.min(score_mat[,1])
  c <- xrange[idx]
  fval_left <- left[idx]
  fval_right <- right[idx]
  if (is.null(xpred) == TRUE){
    results <- list(cut_point=c,fval_left=fval_left,
                    fval_right=fval_right)
  }else{
    ypred <- (xpred <= c) * fval_left + (xpred > c) * fval_right
    results <- list(cut_point=c,fval_left=fval_left,
                    fval_right=fval_right,
                    xpred=xpred,ypred=ypred)
  }
  return(results)
}

# test
# x <- c(1,2,3,4,5,6)
# y <- c(-1,-1,-1,1,1,-1)
# w <- c(1,1,1,1,1,1)
# r <- stump(x,y,w,xpred=2.5)
# r


boosting.stump <- function(x,y,xpred,ytest=NULL,iterations=100,shrinkage=1){
  # Boosting tree using the stump classifier
  # for one dimensional problem.
  
  # @parameters:
  # x: feature vector; n*1; training
  # y: label vector; n*1; training
  # xpred: points to be predicted  
  # ytest: prediction error is given, if provided
  # iterations: total number of base learner
  # shrinkage: further decrease the weight for each learner
  
  # @values:
  # ypred.mat: prediction matrix; n*iternation; cummulative predicition
  #            e.g, the column i represents the predicted value base on 
  #            first i base learners's prediction results
  # pred.error: prediciton error v.s iteration (availabe if ytest provided.) 
  
  w <- rep(1 / length(x), length(x))
  error.mat <- rep(NA,iterations)
  boosting.pred <- matrix(NA,ncol=iterations,nrow=length(xpred))
  for (t in seq_len(iterations)){
    learner.t <- stump(x,y,w,x)
    stump.fitted.t <- learner.t$ypred
    epsilon.t <- t(w) %*% (stump.fitted.t != y)
    alpha.t <- 0.5 * log( (1 - epsilon.t) / epsilon.t) * shrinkage
    w <- w * exp( c(-alpha.t) * y * stump.fitted.t) # update weight
    w <- w / sum(w) # normalize weight
    left <- learner.t$fval_left
    right <- learner.t$fval_right
    cut_point <- learner.t$cut_point
    boosting.pred[,t] <- c(alpha.t) * ((xpred<=cut_point) * left + 
        (xpred>cut_point) * right)
  }
  ypred.mat <- t(apply(boosting.pred,1,cumsum))
  ypred.prob <- apply(ypred.mat,2,function(Fx) 1/(1+exp(-2*Fx)))
  ypred.mat <- sign(ypred.mat)
  if (is.null(ytest) == TRUE){
    results <- list(ypred.prob, 
                    ypred.mat=ypred.mat)
  }else{
    pred.error <- rep(NA,length(xpred))
    for (i in seq_len(ncol(ypred.mat))){
      pred.error[i] <- sum(ytest!=ypred.mat[,i]) / length(ytest)      
    }
    results <- list(ypred.prob=ypred.prob,
                    ypred.mat=ypred.mat,
                    pred.error=pred.error)    
  }
  return(results)
}

n = 100
set.seed(2)
x = runif(n)
y = (rbinom(n , 1 , (sin (4*pi*x)+1)/2)-0.5)*2
r <- boosting.stump(x,y,x,y,iterations=500)
plot(r$pred.error,type="l")


