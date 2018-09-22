# test 1
# For single lambda
library(glmnet)
data(QuickStartExample)
set.seed(1)
cvfit <- cv.glmnet(x, y)
lambda <- cvfit$lambda.min
r <- enet(x,y,alpha=1,lambda)
max(data.frame(r$hatbeta-as.matrix(coef(cvfit,s=cvfit$lambda.min))))

# test 2
# For a sequnece of lambdas
library(glmnet)
r_seq <- enet(x,y,alpha=1, nlambda=100)
fit = glmnet(x, y, alpha = 1, nlambda = 100)
plot(fit, xvar = "lambda", label = TRUE)
matplot((log(r_seq$lambda_seq)), t(r_seq$beta_all[-1,]), 
        type="l",lty = 1,col=1:floor(ncol(x)/2), xlab="Log Lambda", ylab="Coefficients")
lbs_fun <- function(fit, ...) {
  L <- length(fit$lambda_seq)
  xz <- log(fit$lambda_seq[L])-0.1
  yz <- fit$beta_all[, L]
  labs <- 1:dim(fit$beta_all)[1]-1
  text(xz, yz, labels=labs, cex=0.5,...)
}
lbs_fun(r_seq)