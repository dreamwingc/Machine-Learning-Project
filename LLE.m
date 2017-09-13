clear all;close all;clc;
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

TP=1000;
N=784;
L=30;
t_=1;
d=9;
K=18;
train_a=im2double(train_images(:,:,1:5000));
train_lab=train_labels(:,1:5000);
[n,m,z]=size(train_a);
train_a=reshape(train_a,[n*m,z]);
for i=1:10
    t=1;
    t_m=1;
    while(t<=100)
            if (train_lab(:,t_m)==i-1)
                train_l_(1,t_)=train_lab(:,t_m);
                train_i_(:,t_)=train_a(:,t_m);
                t_=t_+1;
                t=t+1;
            end
            t_m=t_m+1;
    end
end

g=randperm(1000);
for i=1:TP
    train_i(:,i)=train_i_(:,g(1,i));
    train_l(1,i)=train_l_(1,g(1,i));
end

for i=1:TP
    dis_=sum((repmat(train_i(:,i),1,TP)-train_i).*(repmat(train_i(:,i),1,TP)-train_i),1);
    dis_(dis_==0)=100000;
    [dis,s]=sort(dis_,'ascend');
    k(:,i)=s(1,1:K);
end  

w=zeros(TP,TP);

for i=1:TP
    for j=1:K
        for k_=1:K
            coe=(train_i(:,i)-train_i(:,k(j,i)))'*(train_i(:,i)-train_i(:,k(k_,i)));
            C(k_,j)=coe;
        end
    end
    w_=(sum(inv(C),1))./sum(sum(inv(C),1),2);
    for k_=1:K
       w(k(k_,i),i)=w_(1,k_);
    end
end

I=zeros(TP,TP);
for i=1:TP
    for r_=1:TP
        if i==r_
            I(r_,i)=1;
        else
            I(r_,i)=0;
        end
    end
end

M=(I-w)'*(I-w);
[V,D]=eig(M);
D(D==0)=[];
[D_,ss]=sort(D,'descend');
V=V';
for i=1:TP
    V_(i,:)=V(ss(1,i),:);
end

for i_=1:d
    y(i_,:)=V_(TP-d+i_,:);
end

h=15;
v=-1+(1-(-1)).*rand(h,d);%hidden weights
weight=-1+(1-(-1)).*rand(10,h);%output weights
u=zeros(h,1);
u_pl=zeros(h,TP);
y_y=zeros(10,1);
y_pj=zeros(10,TP);
c=ones(10,1);
c_=ones(h,1);
eta=0.3;
E_mid=0;
E_=zeros(TP,1);
v_0=zeros(h,1);
w_0=zeros(10,1);
E_final=zeros(10,1);
t_mlp=0;
E_f=0.01;
E_finalt=500;




%while(E_finalt>=0.1)
for hpp=1:2000
    t_mlp=t_mlp+1
    E_mid=0;
    for i=1:TP
    d=zeros(10,1);
    d(train_l(:,i)+1,1)=1;
    o_(:,i)=d;
    net_pl=v*y(:,i)+v_0;
    for x=1:h 
       u(x,1)=1/(1+exp(-net_pl(x,1)));
    end
    u_pl(:,i)=u;  
    net_pj=weight*u+w_0;
    for q=1:10
      y_y(q,1)=1/(1+exp(-net_pj(q,1)));
    end
    y_pj(:,i)=y_y;
    thta=d-y_y;
    delta_w=eta*thta.*y_y.*(c-y_y)*transpose(u);
    weight=weight+delta_w;
    w_0=eta*thta.*y_y.*(c-y_y);
    thta_=weight'*thta;
    delta_v=eta*thta_.*u.*(c_-u)*transpose(y(:,i));
    v=v+delta_v;
    v_0=eta*thta_.*u.*(c_-u);
    E=0.5*sum(thta.^2);
    E_(i,1)=E;
    E_mid=E_mid+E;
    end
    E_finalt=E_mid;
    E_final(t_mlp,1)=E_mid;
end
figure(1)
plot(E_final);
title('Convergence')



TPT=100;
t_=1;
tt_=1;
train_at=im2double(train_images(:,:,5001:6000));
[nt,mt,zt]=size(train_at);
train_at=reshape(train_at,[nt*mt,zt]);
train_la=train_labels(:,5001:6000);
for i=1:10
    t=1;
    t_m=1;
    while(t<=10)
            if (train_la(:,t_m)==i-1)
                train_l_t(1,t_)=train_la(:,t_m);
                train_i_t(:,t_)=train_at(:,t_m);
                t_=t_+1;
                t=t+1;
            end
            t_m=t_m+1;
    end
end

gt=randperm(100);
for i=1:TPT
    train_it(:,i)=train_i_t(:,gt(1,i));
    train_lt(1,i)=train_l_t(1,gt(1,i));
end

for i=1:TPT
    dis_t=sum((repmat(train_it(:,i),1,TP)-train_i).*(repmat(train_it(:,i),1,TP)-train_i),1);
    dis_t(dis_t==0)=100000;
    [dist,st]=sort(dis_t,'ascend');
    kt(:,i)=st(1,1:K);
end  

wt=zeros(TP,TPT);

for i=1:TPT
    for j=1:K
        for k_=1:K
            coet=(train_it(:,i)-train_i(:,kt(j,i)))'*(train_it(:,i)-train_i(:,kt(k_,i)));
            Ct(k_,j)=coet;
        end
    end
    w_t=(sum(inv(Ct),1))./sum(sum(inv(Ct),1),2);
    wttt(:,i)=w_t';
    for k_=1:K
       wt(kt(k_,i),i)=w_t(1,k_);
    end
end   

yt=wt'*y';
yt=yt';

u_=zeros(h,1);
u_pl_=zeros(h,TPT);
y_y_=zeros(10,1);
y_pj_=zeros(10,TPT);
E_u=zeros(10,1);
E_d=zeros(10,1);
Dt=zeros(10,TPT);

for i=1:TPT
    d_=zeros(10,1);
    d_(train_lt(:,i)+1,1)=1;
    Dt(:,i)=d_;
    net_pl_=v*yt(:,i)+v_0;
    for x=1:h 
       u_(x,1)=1/(1+exp(-net_pl_(x,1)));
    end
    u_pl_(:,i)=u_;  
    net_pj_=weight*u_+w_0;
    for q=1:10
      y_y_(q,1)=1/(1+exp(-net_pj_(q,1)));
      if(y_y_(q,1)>=0.5)
          yy_(q,1)=1;
      else yy_(q,1)=0;
      end
    end
    y_pj_(:,i)=yy_;
%     E_test=0.5*sum((d_-yy_).^2);
%     if(E_test==0)
%         E_p=0;
%     else E_p=1;
%     end
%     E_u(train_lt(:,i)+1,1)=E_u(train_lt(:,i)+1,1)+E_p;
%     E_d(train_lt(:,i)+1,1)=E_d(train_lt(:,i)+1,1)+1;
% end
% E_finaltt=E_u./E_d;
end
E_finaltt=sum(abs(Dt-y_pj_),2)/100;
figure(2)
bar(E_finaltt);
axis([0 11 0 1]);
title('The Testing Performance')