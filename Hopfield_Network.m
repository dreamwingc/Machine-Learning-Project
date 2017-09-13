 clear all; close all; clc;

fid = fopen('D:\matlab\train-images.idx3-ubyte','r','ieee-be');
A = fread(fid,4,'uint32');
numberofimages = A(2);
xdim = A(3);
ydim = A(4);
 
train_images = fread(fid, xdim*ydim*numberofimages,'uint8=>uint8');
train_images = reshape(train_images,[xdim, ydim, numberofimages]);
train_images = permute(train_images, [2 1 3]);

% train_images(:,:,i) is a uint8 matrix of size 28x28xi(where i = 1 to 60000)
% train_images are up to 60000 and each one is of size 28x28.

%reading the corresponding training image labels
fid = fopen('D:\matlab\train-labels.idx1-ubyte','r','ieee-be');
a=fread(fid,1,'uint32');
Ntot=fread(fid,1,'uint32');
A=fread(fid);
fclose(fid); 
train_labels=reshape(A,1,Ntot);

%choose 10 digit from 0 to 9
TP=10000;
train_l=train_labels(:,1:TP);
t=0;
train_s=zeros(28,28,10);
while(t<10)
    for i=1:TP
        if(train_l(:,i)==t)
            train_s(:,:,(t+1))=im2double(train_images(:,:,i));
            t=t+1;
        end
    end
end
%geryscale to binary images
for j=1:10
    for a_=1:28
        for b_=1:28
           if(train_s(a_,b_,j)>0)
            train_s(a_,b_,j)=1;
           else
            train_s(a_,b_,j)=-1;
           end;
        end
    end;
end;
train_n=imnoise(train_s,'salt & pepper',0.1);
%show the noise images
% for i=1:1
%     for j=1:10
%         subplot(1,10,j);
%         imshow(train_n(:,:,j));
%     end
% end
% 
% for i=1:size(train_n,3)-1
%     for j =i+1:size(train_n,3)
%         S = dot(squeeze(train_n(:,:,i)), squeeze(train_n(:,:,j)));
%     end
% end
[n,m,z]=size(train_s);
[n_,m_,z_]=size(train_n);
train_s=reshape(train_s,[n*m,z]);
train_n=reshape(train_n,[n_*m_,z_]);
E=zeros(10,1);
E_f=zeros(10,10);

for j=1:10
    for a=1:784
           if(train_n(a,j)>0)
               train_n(a,j)=1;
           else
               train_n(a,j)=-1;
           end;
    end;
end;

w=train_s*(train_s');
for a=1:784
    for b=1:784
        if(a==b)
            w(a,b)=0;
        else
            w(a,b)=w(a,b);
        end
    end
end

for i_=1:5
    E=zeros(10,1);
    Net_j=w*train_n;
    for j=1:10
        for a=1:784
            if(Net_j(a,j)>0)
                train_n(a,j)=1;
            else
                if(Net_j(a,j)==0)
                    train_n(a,j)=train_n(a,j);
                else
                    train_n(a,j)=-1;
                end
            end
        end
    end
    en(i_,:)=(-0.5)*sum((Net_j.*train_n),1);
    for j=1:10
        for a=1:784
            if(train_n(a,j)==train_s(a,j))
                E(j,1)=E(j,1);
            else
                E(j,1)=E(j,1)+1;
            end
        end
        E_(j,1)=E(j,1)/784;
    end
    E_f(:,i_)=E_;
end
train_n=reshape(train_n,[28,28,10]);
for i=1:1
    for j=1:10
        subplot(1,10,j);
        imshow(train_n(:,:,j));
    end
end

for i=1:size(train_n,3)-1
    for j =i+1:size(train_n,3)
        S = dot(squeeze(train_n(:,:,i)), squeeze(train_n(:,:,j)));
    end
end