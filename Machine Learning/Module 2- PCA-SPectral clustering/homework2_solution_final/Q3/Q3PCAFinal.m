clear
T = readtable('food-consumption.txt','Delimiter','\t','HeaderLines',1,'ReadRowNames',true);
data = table2array(T);                      
countrynames = T.Properties.RowNames;        
foodnames = T.Properties.VariableNames;
Anew= data; 
[m,n]=size(Anew);
stdA = std(Anew, 1, 1); 
Anew = Anew * diag(1./stdA); 
Anew = Anew'; 
mu=sum(Anew,2)./m;
xc = bsxfun(@minus, Anew, mu); 
                 
C = xc * xc' ./ m; 
k = 2; 
[W, S] = eigs(C, k); 

dim1 = W(:,1)' * xc ./ sqrt(S(1,1));
dim2 = W(:,2)' * xc ./ sqrt(S(2,2));

plot(dim1,dim2,'o'); hold on  
text(dim1 ,dim2,countrynames,'FontSize',7,'VerticalAlignment','bottom');  

z = zeros(1,20);
plot([z;W(:,1)'],[z;W(:,2)'],'r');  
text(W(:,1),W(:,2),foodnames,'FontSize',7,'VerticalAlignment','bottom','HorizontalAlignment','right','color','red');
grid on; axis equal
xlabel('c_1'); ylabel('c_2')