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

TP=10000;
L=50;
train_d=im2double(train_images(:,:,1:TP));
[n,m,z]=size(train_d);
train_d=reshape(train_d,[n*m,z]);
train_l=train_labels(:,1:TP);
mean=zeros(L,1);
e_m=zeros(TP,L);
num=zeros(TP,L);
mu=zeros(784,L);
dis_n=zeros(TP,L);
dis=zeros(L,1);
dis_f=zeros(L,TP);
train_sa=zeros(784,L);
w=-0.5+1*rand(10,L);
seg_i=zeros(784,1);
n_mu=zeros(784,TP);
u=zeros(L,1);
seg_=0;
delta_w=zeros(10,L);
eta=0.2;
y=zeros(10,1);
v_=zeros(10,1);
mu_=ones(784,L);
seg_f=zeros(L,1);
m_=zeros(L,1);
E_p=3250;
t=0;
%hid=randi([1,TP],1,400);
hid=randperm(L);
mu=train_d(:,hid');
while(sum(sum(mu-mu_))~=0)
    mu_=mu;
%     E_=0;
%     E_f=0;
    mean_d=zeros(L,1);
    train_s=zeros(784,L);
    num=zeros(TP,L);
for i=1:TP
    for a=1:L
    dis(a,1)=sum((train_d(:,i)-mu(:,a)).^2);
    end
    [w_,x]=min(dis);
    mean_d(x,1)=mean_d(x,1)+1;
    num(mean_d(x,1),x)=i;
end
for a=1:L
    for i=1:TP
        if(num(i,a)==0)
            train_s(:,a)=train_s(:,a);
        else
            train_s(:,a)=train_s(:,a)+train_d(:,num(i,a));
        end;
    end
    train_sa(:,a)=train_s(:,a)/mean_d(a,1);
    mu(:,a)=train_sa(:,a);
%     for i=1:TP   
%        if(num(i,a)==0)
%            E=E;
%        else
%            E=sum((train_d(:,num(i,a))-mu(:,a)).^2);
%        end
%        E_=E_+E;
%     end
%     E_f=E_f+E_;
end
% E_ff(u)=E_f;
end

for a=1:L
    seg_i=0;
        for i=1:TP
%         seg_i=seg_i+((train_d(:,i)-mu(:,a)).^2);
          if(num(i,a)==0)
              seg_i=seg_i;
          else
              seg_i=seg_i+(train_d(:,num(i,a))-mu(:,a)).^2;
          end
        end
    seg_k=sum(sqrt(seg_i),1)/784;
%    seg_f(a,1)=seg_k;
    seg_=seg_+seg_k;
end
seg=seg_/L;
% for i=1:TP
%     [q,p]=find(num==i);
%     n_mu(:,i)=mu(:,p);
% end

while(E_p>1900)
    t=t+1
    E_p=0;
    for i=1:TP
        d=-ones(10,1);
        d(train_l(:,i)+1,1)=1;
        o_(:,i)=d;
    for a=1:L
        m_(a,1)=sum((train_d(:,i)-mu(:,a)).^2);
    end
    u=(1/(2*(seg^2)))*exp(-(m_)/(2*(seg^2)));
    v_=w*u;
    for g=1:10;
    if(v_(g,1)<0)
    y(g,1)=-1;
    else y(g,1)=1;
    end;
    end
    y_(:,i)=y;
    delta_w=eta*(d-y)*u';
    w=w+delta_w;
    E_pp=(1/10)*sum((d-y).^2);
    E_p=E_p+E_pp;
    end
    E_f(t)=E_p;
end

TP_=13000;
train_t=im2double(train_images(:,:,10001:TP_));
[f_,i_,z]=size(train_t);
train_t=reshape(train_t,[f_*i_,z]);%3D array to 2D array, input 
train_lt=train_labels(:,10001:TP_);%standard
y_t=zeros(10,1);
v_t=zeros(10,1);
E_test=0;
E_u=zeros(10,1);
E_d=zeros(10,1);
u_t=zeros(L,3000);
seg_t=0;
% l_=zeros(10,101);
% o_=zeros(10,101);

% for a=1:L
%     seg_it=0;
%         for r_=1:3000
%         seg_it=seg_it+((train_t(:,r_)-mu(:,a)).^2);
% %           if(num(i,a)==0)
% %               seg_i=seg_i;
% %           else
% %               seg_i=seg_i+(train_d(:,num(i,a))-mu(:,a)).^2;
% %           end
%         end
%     seg_kt=sum(sqrt(seg_it),1)/784;
% %    seg_f(a,1)=seg_k;
%     seg_t=seg_t+seg_kt;
% end
% segt=seg_t/L;

for r_=1:3000;
    d_t=-ones(10,1);
    d_t(train_lt(:,r_)+1)=1;
    l_t(:,r_)=d_t;
    for a=1:L
        u_t(a,r_)=(1/(2*(seg^2)))*exp(-(sum((train_t(:,r_)-mu(:,a)).^2))/(2*(seg^2)));
    end
    v_t=w*u_t(:,r_);
    for g_=1:10;
    if(v_t(g_,1)<0)
    y_t(g_,1)=-1;
    else y_t(g_,1)=1;
    end;
    o_t(:,r_)=y_t;
    end;
    E_test=0.5*sum((d_t-y_t).^2);
    if(E_test==0)
        E_p=0;
    else E_p=1;
    end
    E_u(train_lt(:,r_)+1,1)=E_u(train_lt(:,r_)+1,1)+E_p;
    E_d(train_lt(:,r_)+1,1)=E_d(train_lt(:,r_)+1,1)+1;
end
E_finalt=E_u./E_d;