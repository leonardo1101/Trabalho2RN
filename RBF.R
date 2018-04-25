
    # Implementacao simples da RBF
    # http://www.di.fc.ul.pt/~jpn/r/rbf/rbf.html
    
    # Cria um dataframe com 2 colunas, 100 observacoes
    # Esses serao os dados de treinamento

    desclassificadorIsis <- function(classe){
        if(classe == 'setosa'){ 
            return(-1)
        }
        if(classe == 'versicolor'){
            return(0)
        }
        if(classe == 'virginica'){
            return(1)
        }
    }
    target <- function(x1, x2) {
        2*(x2 - x1 + .25*sin(pi*x1) >= 0)-1
    }


    # Funcao RBF com o LSM com pseudo-inversa
    # Retorna um modelo RBF dado:
    # * observacoes x1...xN do dataset
    # * valor de saida de cada observacao
    # * numero de centros
    # * valor gama para a funcao Gaussiana
    # Precisa so pacote corpcor para a pseudo inversa

    library(corpcor)
    library(SDMTools)
    rbf <- function(X, Y, K=10, gama=1.0) {
        N <- dim(X)[1] # numero de observacoes
        ncols <- dim(X)[2] # numero de variaveis
        repeat {
            km <- kmeans(X, K) # agrupa os dados em K grupos
            if (min(km$size)>0) # nao pode haver grupos vazios
            break
        }
        
        mus <- km$centers # centros dos grupos (medias)

    # Calcula as saidas das Gaussianas
        Phi <- matrix(rep(NA,(K+1)*N), ncol=K+1) # Vai armazenar todas as saidas mais o bias
        for (lin in 1:N) {
        
            Phi[lin,1] <- 1
            # coluna do bias
            for (col in 1:K) {
                Phi[lin,col+1] <- exp( (-1/(2*gama*gama)) * sum((X[lin,]-mus[col,])*(X[lin,]-mus[col,])) )

                #Phi[lin,col+1] <- exp( -gama * sum((X[lin,]-mus[col,])*(X[lin,]-mus[col,])))
                #Phi[lin,col+1] <- exp( -gama * norm(as.matrix(X[lin,]-mus[col,]),"F")^2 )
            }
        }
        # Calcula os pesos com a pseudo inversa -> w = inversa(t(Phi) * Phi) * t(Phi) * Y
        # Encontra os pesos fazendo a inversa
        # %*% é para multiplicacao de matrizes
        w1 <- pseudoinverse(t(Phi) %*% Phi) %*% t(Phi) %*% Y[,1]
        w2 <- pseudoinverse(t(Phi) %*% Phi) %*% t(Phi) %*% Y[,2]
        w3 <- pseudoinverse(t(Phi) %*% Phi) %*% t(Phi) %*% Y[,3]
        w <- cbind(w1,w2)
        w <- cbind(w,w3)
        return(list(pesos=w, centros=mus, gama=gama))
}
     # retorna o modelo RBF
    # Treina o modelo
    #modelo

    # Implementacao da funcao para predicao
    rbf.predict <- function(modelo, X, classification=FALSE) {
        gama <- modelo$gama
        centros <- modelo$centros
        w <-  modelo$pesos
        N <-  nrow(X)
        
        # numero de observacoes
        pred1 <- rep(w[1,1],N)
        pred2 <- rep(w[1,2],N)
        pred3 <- rep(w[1,3],N)
        pred <- cbind(pred1,pred2)
        pred <- cbind(pred,pred3)
        
        
        # inicia com o peso do bias ja que a entrada associada eh 1
        for (j in 1:N) {
            # Predicao para o ponto xj
            for (k in 1:length(centros[,1])) {
                # o peso para o centro[k] é dado por w[k+1] porque w[1] eh o bias
                pred[j,1] <- pred[j,1] + w[k+1,1] * exp( (-1/(2*gama*gama)) *
                sum((X[j,]-centros[k,])*(X[j,]-centros[k,])) )
                pred[j,2] <- pred[j,2] + w[k+1,2] * exp( (-1/(2*gama*gama)) *
                sum((X[j,]-centros[k,])*(X[j,]-centros[k,])) )
                pred[j,3] <- pred[j,3] + w[k+1,3] * exp( (-1/(2*gama*gama)) *
                sum((X[j,]-centros[k,])*(X[j,]-centros[k,])) )
            #pred[j]<-pred[j]+w[k+1]*exp(-gama*sum((X[j,]-centros[k,])*(X[j,]-centros[k,])))
            #pred[j]<-pred[j]+w[k+1]*exp(-gama*norm(as.matrix(X[j,]-centros[k,]),"F")^2)
            }
        }
        # Se for classificacao, aplica a funcao sinal em cada pred
       
        if (classification) {
            previ <- matrix(0,nrow=nrow(X),ncol=3)
        for(i in 1:nrow(X)){
            for(j in 1:3){
                if(sign(pred[i,j]) == 1)
                    previ[i,j]<- 1
                else
                    previ[i,j]<- 0
            }
        }
        
        return(previ)
        }
        
        return(pred)
    }
    
    target <- function(x1, x2) {
        2*(x2 - x1 + .25*sin(pi*x1) >= 0)-1
    }
    convertIris <- function(){
        m <- matrix (0, nrow = 150, ncol = 7)
        for (i in 1:150){
            for(j in 1:5){
                if(j < 5 ){            
                    m[i,j] <- iris[i,j]
                }else{
                    if(iris[i,j]=='setosa'){
                        m[i,j] <- 1
                        m[i,j + 1] <- 0
                        m[i,j + 2] <- 0
                    }
                    if(iris[i,j]=='versicolor'){
                        m[i,j] <- 0
                        m[i,j + 1] <- 1
                        m[i,j + 2] <- 0
                    }
                    if(iris[i,j]=='virginica'){
                        m[i,j] <- 0
                        m[i,j + 1] <- 0
                        m[i,j + 2] <- 1
                    }
                }
            }
        }
        return(m)
    }

    minhaIris <- convertIris()
    
    for(i in 1:10){
        
        a <- as.integer(i* 15 + 1)
        b <- as.integer(1 + (i -1) * 15)
        c<- as.integer(i* 15)
        d <- as.integer((i -1) * 15)
        teste <- minhaIris[b:c,]
        if(i != 10 && i != 1){
            treinamento <- minhaIris[1: d,]
            treinamento <- rbind(treinamento,minhaIris[a : 150,])
        }else{
            if(i == 1)
                treinamento <- minhaIris[a: 150,]
            else
                treinamento <- minhaIris[1: d,]
        }

        X <- treinamento[,1:4]
        Y <- treinamento[,5:7]
        
        modelo <- rbf(X,Y)
        
        X.out <- teste[,1:4]
        Y.out <- teste[,5:7]
         
        rbf.pred <- rbf.predict(modelo, X.out, classification=TRUE)
        soma <- 0 
        for(i in 1:15){
            if(rbf.pred[i,1] != Y.out[i,1] || rbf.pred[i,2] != Y.out[i,2] || rbf.pred[i,3] != Y.out[i,3])
                soma <- soma + 1
        }
        matrixC <-confusion.matrix(Y.out, rbf.pred, threshold = 0.5)

        print("Erro:")
        
        print(soma/15)
        
        print("Matriz confusão:")
        print(matrixC)
    }

    # Mostrando os resultados graficamente
    #plot(X.out$x1, X.out$x2, col=Y.out+3, pch=0)
    #points(X.out$x1, X.out$x2, col=rbf.pred+3, pch=3)
    #points(modelo$centros, col="black", pch=19) # draw the model centers
    #legend("topleft",c("true value","predicted"),pch=c(0,3),bg="white")
