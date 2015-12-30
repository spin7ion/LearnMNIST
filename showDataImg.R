showDataImg <- function(data, labels, n){ #Just show image and print label
  f <- function(m) t(m)[,nrow(m):1]#rotates 
  image( f(t(matrix(data[n, ], ncol=28, nrow=28))), Rowv=28, Colv=28, col = heat.colors(256),  margins=c(5,10), xlab=paste("Class label:", labels[n]))
  print(paste("Class label:", labels[n]))
}

