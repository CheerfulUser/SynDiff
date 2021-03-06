function [ Tlpp, Y ] = computeOneTransMetricKnownMap( X,Ymap )
%Using a previous map and population (map) of known transits
%Compute the lpp-knn transit metric on one folded bin light curves
%return the reduced dimensions
%return the value of the transit metric.
% X is a vector you want to map. (1xN)

%Apply the LPP map to the old sample
%[Ygood]=maplle_oos(map.X,map.Ymap,map.nDim);
%[Zgood]=maplle_oos(map.X,map.Zmap,map.nDim);
%Zorig=map.Zmap.mapped;

%Apply the LPP map for the out of sample
[Y]=maplle_oos(X,Ymap.mapping,Ymap.nDim);

%The original mapped vectors
Yorig=Ymap.mapped;

%x are known transits
%y are those that need to be classified
x=Yorig(Ymap.knnGood,:);  

%[~,dy]=knnsearch(x,Y,'k',Ymap.knn,'distance','minkowski','p',3);
[~,dy]=kNearestNeighbors(x,Y,Ymap.knn)
Tlpp=mean(dy');

end

