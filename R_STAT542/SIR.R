#!/usr/bin/env R
# -*- coding: utf-8 -*-
# @Date    : 2017-12-01 11:11:56
# @Author  : Roth (rothdyt@gmail.com)
# @Version : 0.9

SIR <- function(X, Y, slices=10){
    # perform sliced inverse regression to achieve dimension reduction
    # choose first K columns of matrxi `directions` returned by SIR
    # will give you the reduced n by k design matrix.

    # @ parameters:
    #   X: n by p design matrix
    #   Y: n by 1 response vector
    #   slices: number of slices in partioning Y 
    # @ values:
    #   M: p by p matrix; Used to find raw directions
    #   raw.eigen.vectors: p by p matrix; eigen vectors of  M
    #   directions: truelinear transformation of raw.eigen.vectors
    #   eigenvalues: eigen values of M
    #   slice.info: record information of how the partition is conducted

    X.row <- nrow(X); X.col <- ncol(X)
    X.centerd <- scale(X, center=T, scale=F)
    svd <- eigen(cov(X) * (n-1) / n) # use 1/n covariance  
    Gamma <- svd$vectors
    Sigma_inv <- diag(1/svd$values)
    hat_inv_Sigma_sqrt <- Gamma %*% sqrt(Sigma_inv) %*% t(Gamma)
    Z <- X.centerd %*% hat_inv_Sigma_sqrt
    Z <- Z[order(Y),]
    Y <- Y[order(Y)]
    idx <- split(seq_len(X.row), 
                 rep(1:slices, each=ceiling(X.row/slices), length.out=X.row))
    Z_h <- matrix(NA, ncol=X.col, nrow=slices)
    slice_size <- rep(NA, slices)
    for (i in seq_len(slices)){
      Z_h[i,] <- colMeans(Z[idx[[i]], ,drop=FALSE])
      slice_size[i] <- length(idx[[i]])
    }
    Mat <- t(Z_h) %*% apply(Z_h, 2, "*", slice_size)/sum(slice_size)
    D <- eigen(Mat)
    raw.evectors <- D$vectors
    eigen.values <- D$values
    norm_vec <- function(x) sqrt(sum(x^2))
    directions <- hat_inv_Sigma_sqrt %*% raw.evectors
    # nomralize by column
    normalizer <- t(1 / apply(directions, 2, norm_vec) %*% t(rep(1,p))) 
    directions <- directions * normalizer
    slice.info <- list(nslices=slices, slice.size=slice_size)
    results <- list(directions=directions, 
                    Eigenvalues=eigen.values,
                    slice.info=slice.info,
                    M=Mat,
                    raw.eigen.vectors=raw.evectors)
    return(results)
}



# how dr package compute M matrix
# H = 5
# X.row <- nrow(X); X.col <- ncol(X)
# X.center <- scale(X, center=T, scale=F)
# sweights <- sqrt(rep(1,n))
# qr <- qr(scale(apply(X, 2, function(a, sweights) a * sweights, 
#                      sweights), center = TRUE, scale = FALSE))
# Zdr <- sqrt(n) * qr.Q(qr)[, 1:p]
# Zdr <- Zdr[order(Y),]
# Y <- Y[order(Y)]
# idx <- split(seq_len(X.row), 
#              rep(1:H, each=ceiling(X.row/H), length.out=X.row))
# Z_h <- matrix(NA, ncol=X.col, nrow=H)
# slice_size <- rep(NA, H)
# for (i in seq_len(H)){
#   Z_h[i,] <- colMeans(Zdr[idx[[i]],])
#   slice_size[i] <- length(idx[[i]])
# }
# Matdr <- t(Z_h) %*% apply(Z_h, 2, "*", slice_size)/sum(slice_size)

# test example 1

library(dr)
set.seed(1234)
n = 300; p = 10
X = matrix(rnorm(n*p), n, p)
b = matrix(c(1, 1, rep(0, p-2)))
Y = 0.125*(X %*% b)^3 + 0.5*rnorm(n)
fit.sir = dr(Y~., data = data.frame(X, Y), method = "sir", nslices=5, numdir=p)
myfit = SIR(X,Y,slices=5)
myfit$directions[,c(1,2)]
fit.sir$evectors[,c(1,2)]
myfit$Eigenvalues
plot(myfit$Eigenvalues)

myB <- myfit$directions[,1]
trueB <- cbind(b1,b2)
trueB %*% solve(t(trueB) %*% trueB) %*% t(trueB)


