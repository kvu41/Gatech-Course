# Load data
X1 = read.csv("X1.csv", header=FALSE)
matplot(t(X1), type = "l", xlab = "data points", ylab = "power signal")

# Sample average signal 
mean.signal = apply(X1, 2, mean)
plot(mean.signal, xlab = "data points", ylab = "power signal", type = "l")

# cubic splines
X = 1:51
k = seq(0,51,length.out = 8)
k = k[2:7]
h1 = rep(1,length(X))
h2 = X
h3 = X^2
h4 = X^3
h5 = (X-k[1])^3
h5[h5 <= 0] = 0
h6 = (X-k[2])^3
h6[h6 <= 0] = 0
h7 = (X-k[3])^3
h7[h7 <= 0] = 0
h8 = (X-k[4])^3
h8[h8 <= 0] = 0
h9 = (X-k[5])^3
h9[h9 <= 0] = 0
h10 = (X-k[6])^3
h10[h10 <= 0] = 0
H = cbind(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10)
B=solve(t(H)%*%H)%*%t(H)%*%mean.signal
cubic = H%*%B
plot(mean.signal,xlab = "data points", ylab = "power signal", col = "black",pch=20)   
lines(cubic,col = "red",lwd = 1)
#mse
mse1 = sum((mean.signal-cubic)^2)/51
#mse leave-1-out CV
x = 1:51
y = mean.signal
er = rep(0, length(y))
for(i in 1:51)
{
  Y1 = y[-i]
  H1 = H[-i,]
  B1 = solve(t(H1)%*%H1)%*%t(H1)%*%Y1
  er[i] = y[i]-H[i,]%*%B1
}
mse1cv = sum(er^2)/51

# B-splines
#install.packages("splines")
library(splines)
X = 1:51
knots = seq(0,51,length.out = 9)
B = bs(X, knots = knots, degree = 2,intercept = FALSE)[,1:10]
bspline = B%*%solve(t(B)%*%B)%*%t(B)%*%mean.signal
plot(mean.signal,xlab = "data points", ylab = "power signal", col = "black", pch = 20)   
lines(bspline,col = "red",lwd = 1)
#mse
mse2 = sum((mean.signal-bspline)^2)/51
#mse leave-1-out CV
x = 1:51
y = mean.signal
er = rep(0, length(y))
for(i in 1:51)
{
  Y1 = y[-i]
  B1 = B[-i,]
  bs1 = solve(t(B1)%*%B1)%*%t(B1)%*%Y1
  er[i] = y[i]-B[i,]%*%bs1
}
mse2cv = sum(er^2)/51

# Smoothing splines
n = 51
allspar = seq(0,1,length.out = 1000)
p = length(allspar)
RSS = rep(0,p)
df = rep(0,p)
for(i in 1:p)
{
  yhat = smooth.spline(mean.signal, df = n, spar = allspar[i])
  df[i] = yhat$df
  yhat = yhat$y
  RSS[i] = sum((yhat-mean.signal)^2)
}
GCV = (RSS/n)/((1-df/n)^2)
plot(allspar,GCV,type = "l", lwd = 3)
spar = allspar[which.min(GCV)]
points(spar,GCV[which.min(GCV)],col = "red",lwd=5)
smooth = smooth.spline(mean.signal, df = n, spar = spar)
smooth = smooth$y
plot(mean.signal,xlab = "data points", ylab = "power signal", col = "black", pch=20)   
lines(smooth,col = "red",lwd = 1)
#mse
mse3 = sum((mean.signal-smooth)^2)/51
#mse leave-1-out CV
x = 1:51
y = mean.signal
er = rep(0, length(y))
for(i in 1:51)
{
  Y1 = y[-i]
  smooth1 = smooth.spline(Y1, df = n-1, spar = spar)
  ypred = predict(smooth1, i)
    er[i] = y[i]-ypred$y
}
mse3cv = sum(er^2)/51

#Kernel regression
x = 1:51
y = mean.signal
kerf = function(z){exp(-z*z)/sqrt(2*pi)}
h1=seq(0.001,0.1,0.0001)
er = rep(0, length(y))
mse = rep(0, length(h1))
for(j in 1:length(h1))
{
  h=h1[j]
  for(i in 1:length(y))
  {
    X1=x[-i];
    Y1=y[-i];
    z=kerf((x[i]-X1)/h)
    yke=sum(z*Y1)/sum(z)
    er[i]=y[i]-yke
  }
  mse[j]=sum(er^2)
}
plot(h1,mse,type = "l")
h = h1[which.min(mse)]
points(h,mse[which.min(mse)],col = "red", lwd=5)
N=1000
xall = seq(1,51,length.out = N)
f = rep(0,N);
for(k in 1:N)
{
  z=kerf((xall[k]-x)/h)
  f[k]=sum(z*mean.signal)/sum(z);
}
plot(xall,f,xlab = "data points", ylab = "power signal",type = "l", col = "red", lwd=1)   
points(x,y,pch=20)
## mse
kernel = rep(0,51)
for(i in 0:49)
{
  kernel[i+1] = f[1+i*20]
}
kernel[51] = f[N]
mse4 = sum((mean.signal-kernel)^2)/51
#mse leave-1-out CV
mse4cv = mse[which.min(mse)]/51
