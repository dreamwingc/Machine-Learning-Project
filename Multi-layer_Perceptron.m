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

TP=10000;%pictures for training
train_d=im2double(train_images(:,:,1:TP));
[n,m,z]=size(train_d);
train_d=reshape(train_d,[n*m,z]);%input
train_l=train_labels(:,1:TP);
y=zeros(10,1);%compute output initialize
h=zeros(10,1);%hidden nodes initialize
v=-1+(1-(-1)).*rand(10,n*m);%hidden weights
w=-1+(1-(-1)).*rand(10,10);%output weights
u=zeros(10,1);
u_pl=zeros(10,10000);
y=zeros(10,1);
y_pj=zeros(10,10000);
c=ones(10,1);
eta=0.9;
E_mid=0;
E_=zeros(10000,1);
v_0=zeros(10,1);
w_0=zeros(10,1);
E_final=zeros(10,1);
t=0;
E_f=0.01;
E_finalt=500;

%hidden nodes compute
while(E_finalt>=450)
    t=t+1;
    E_mid=0;
    for i=1:TP
    d=zeros(10,1);
    d(train_l(:,i)+1,1)=1;
    net_pl=v*train_d(:,i)+v_0;
    for x=1:10 
       u(x,1)=1/(1+exp(-net_pl(x,1)));
    end
    u_pl(:,i)=u;  
    net_pj=w*u+w_0;
    for q=1:10
      y(q,1)=1/(1+exp(-net_pj(q,1)));
    end
    y_pj(:,i)=y;
    thta=d-y;
    delta_w=eta*thta.*y.*(c-y)*transpose(u);
    w=w+delta_w;
    w_0=eta*thta.*y.*(c-y);
    thta_=w'*thta;
    delta_v=eta*thta_.*u.*(c-u)*transpose(train_d(:,i));
    v=v+delta_v;
    v_0=eta*thta_.*u.*(c-u);
    E=0.5*sum(thta.^2);
    E_(i,1)=E;
    E_mid=E_mid+E;
    end
    E_finalt=E_mid;
    E_final(t,1)=E_mid;
    plot(E_final);
end

%Testing
TTP=13000;%pictures for training
test_d=im2double(train_images(:,:,10001:TTP));
[l,k,o]=size(test_d);
test_d=reshape(test_d,[l*k,o]);%input
test_l=train_labels(:,10001:TTP);
u_=zeros(10,1);
u_pl_=zeros(10,3000);
y_=zeros(10,1);
y_pj_=zeros(10,3000);
E_u=zeros(10,1);
E_d=zeros(10,1);
D=zeros(10,3000);
for i=1:3000
    d_=zeros(10,1);
    d_(test_l(:,i)+1,1)=1;
    D(:,i)=d_;
    net_pl_=v*test_d(:,i)+v_0;
    for x=1:10 
       u_(x,1)=1/(1+exp(-net_pl_(x,1)));
    end
    u_pl_(:,i)=u_;  
    net_pj_=w*u_+w_0;
    for q=1:10
      y_(q,1)=1/(1+exp(-net_pj_(q,1)));
      if(y_(q,1)>=0.5)
          y_(q,1)=1;
      else y_(q,1)=0;
      end
    end
    y_pj_(:,i)=y_;
    E_test=0.5*sum((d_-y_).^2);
    if(E_test==0)
        E_p=0;
    else E_p=1;
    end
    E_u(test_l(:,i)+1,1)=E_u(test_l(:,i)+1,1)+E_p;
    E_d(test_l(:,i)+1,1)=E_d(test_l(:,i)+1,1)+1;
end
E_finaltt=E_u/E_d;
bar(E_finaltt);