function [ class, centroid ] = mykmedoids( pixels, K )
%
% Your goal of this assignment is implementing your own K-medoids.
% Please refer to the instructions carefully, and we encourage you to
% consult with other resources about this algorithm on the web.
%
% Input:
%     pixels: data set. Each row contains one data point. For image
%     dataset, it contains 3 columns, each column corresponding to Red,
%     Green, and Blue component.
%
%     K: the number of desired clusters. Too high value of K may result in
%     empty cluster error. Then, you need to reduce it.
%
% Output:
%     class: the class assignment of each data point in pixels. The
%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
%     of class should be either 1, 2, 3, 4, or 5. The output should be a
%     column vector with size(pixels, 1) elements.
%
%     centroid: the location of K centroids in your result. With images,
%     each centroid corresponds to the representative color of each
%     cluster. The output should be a matrix with K rows and
%     3 columns. The range of values should be [0, 255].
%     
%
% You may run the following line, then you can see what should be done.
% For submission, you need to code your own implementation without using
% the kmeans matlab function directly. That is, you need to comment it out.

% 	[class, centroid] = kmeans(pixels, K);
  row=size(pixels,1);
    randoms=randperm(row,K)
    centroid=pixels(randoms,:);
    class=ceil(rand(row,1)*K);
    change=true;
    counter=0
    
    while (counter<10 & change)
        counter=counter+1
        for i =1:row
            mindist=sum(abs(centroid(1,:)-pixels(i,:)));
            minindex=1;
            for j = 1:K
                dist=sum(abs(centroid(j,:)-pixels(i,:)));
                if (dist<mindist)
                    mindist=dist;
                    minindex=j;
                end
            end 
            if minindex~=class(i,1)
                change=true;
                class(i,1)=minindex;
            end
        end
        for j =1:K
            clustersample=find(class(:,1)==j);
            if size(clustersample)>0
            avg=mean(pixels(clustersample,:));
            mindist=sum(abs(avg-pixels(clustersample(1,1),:)));
            minindex=1;
            for i =clustersample'
                dist=sum(abs(avg-pixels(i,:)));
                if (dist<mindist)
                    mindist=dist;
                    minindex=i;
                end
            end
            centroid(j,:)=pixels(minindex,:);
            end
        end
    end
end