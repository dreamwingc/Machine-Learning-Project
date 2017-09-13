clear all;
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

TP=100;
N=784;
L=30;
t_=1;
tt_=1;
train_a=im2double(train_images(:,:,1:500));
[n,m,z]=size(train_a);
train_a=reshape(train_a,[n*m,z]);
for i=1:10
    t=1;
    t_m=1;
    while(t<=10)
            if (train_labels(:,t_m)==i-1)
                train_l(1,t_)=train_labels(:,t_m);
                train_i(:,t_)=train_a(:,t_m);
                t_=t_+1;
                t=t+1;
            end
            t_m=t_m+1;
    end
end
V_=zeros(N,N);  
w_m=zeros(L,10);

M=(1/TP)*sum(train_i,2);
train_x=train_i-repmat(M,1,100);
C=train_x*train_x';
[V,D]=eig(C);
D=sum(D,1);
D=sort(D,'descend');
B=1:N;
B=sort(B,'descend');
for j=1:L
    V_(:,j)=V(:,B(1,j));
end
m_r=sum(D(1,1:L),2)/sum(D,2);
V_=V_(:,1:L);
w=V_'*train_x;

for a=1:10
    w_m(:,a)=(1/10)*sum(w(:,(10*(a-1)+1):(10*(a-1)+10)),2);
end


for a=1:10
    for b=1:10
    sd_(b,1)=((1/L)*sum((w(:,(10*(a-1)+b))-w_m(:,a)).*(w(:,(10*(a-1)+b))-w_m(:,a)),1));
    end
    sd(a,1)=max(sd_);
end

d=sd;

train_at=im2double(train_images(:,:,501:1000));
[n_t,m_t,z_t]=size(train_at);
train_at=reshape(train_at,[n_t*m_t,z_t]);
train_lt_=train_labels(:,501:1000);
for i=1:10
    t_t=1;
    t_mt=1;
    while(t_t<=10)
            if (train_lt_(:,t_mt)==i-1)
                train_lt(:,tt_)=train_lt_(:,t_mt);
                train_it(:,tt_)=train_at(:,t_mt);
                tt_=tt_+1;
                t_t=t_t+1;
            end
            t_mt=t_mt+1;
    end
end
num=zeros(10,1);
train_f=zeros(1,TP);
train_xt=train_it-repmat(M,1,100);
wt=V_'*train_xt;

for i_=1:TP
    Dl=(1/L)*sum((w_m-repmat(wt(:,i_),1,10)).*(w_m-repmat(wt(:,i_),1,10)),1);
    [x,y]=min(Dl);
    x_(i_,1)=x;
    if(x<=d(y,1))        
        train_f(1,i_)=y-1;
    else
        train_f(1,i_)=-1;
    end
end

for i_=1:TP
    if(train_f(1,i_)~=train_lt(1,i_))
        num(train_lt(1,i_)+1,1)=num(train_lt(1,i_)+1,1)+1;
    else
        num=num;
    end
end
E=num/10;
        
bar(E)
axis([0 11 0 1]);
title('Testing Performance')