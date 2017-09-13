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

%train_labels(i) - 60000x1 uint8 vector
TP=300;
%geryscale to binary images
for i=1:TP;
    for a=1:28;
        for b=1:28;
            if(train_images(a,b,i)>0)
                train_images(a,b,i)=1;
            end;
        end;
    end;
end;
train_t=im2double(train_images(:,:,1:TP));
[n,m,z]=size(train_t);
train_t=reshape(train_t,[n*m,z]);%3D array to 2D array, input 
train_l=train_labels(:,1:TP);%standard
weights=randn(10,n*m);%initialize
w_0=0.01;
v=zeros(10,1);
y=zeros(10,1);
eta=0.7;
e=0.05;
E_final=0.05;
E_every=0;
E=0;
l=zeros(10,300);
o=zeros(10,300);
w=zeros(10,300);

while(E_final>=e)
    E=0;
    for i=1:z;
    d=-ones(10,1);
    d(train_l(:,i)+1)=1;
    l(:,i)=d;
    v=weights*train_t(:,i)+w_0;
    w(:,i)=v;
    for g=1:10;
    if(v(g,1)<0)
    y(g,1)=-1;
    else y(g,1)=1;
    end;
    end;
    delta_w=eta*(d-y)*transpose(train_t(:,i));
    weights=weights+delta_w;
    o(:,i)=y;
    E_every=0.5*sum((d-y).^2);
    E=E+E_every;
    end;
    E_final=E;
end;

TP_=400;
%geryscale to binary images
for r=300:TP_;
    for a=1:28;
        for b=1:28;
            if(train_images(a,b,r)>0)
                train_images(a,b,r)=1;
            end;
        end;
    end;
end;
train_test=im2double(train_images(:,:,300:TP_));
[u,q,z]=size(train_test);
train_test=reshape(train_test,[u*q,z]);%3D array to 2D array, input 
train_lt=train_labels(:,300:TP_);%standard
y_test=zeros(10,1);
v_=zeros(10,1);
E_test=0;
E_u=zeros(10,1);
E_d=zeros(10,1);
l_=zeros(10,101);
o_=zeros(10,101);
for r=1:101;
    d_test=-ones(10,1);
    d_test(train_lt(:,r)+1)=1;
    l_(:,r)=d_test;
    v_=weights*train_test(:,r)+w_0;
    for g_=1:10;
    if(v_(g_,1)<0)
    y_test(g_,1)=-1;
    else y_test(g_,1)=1;
    end;
    o_(:,r)=y_test;
    end;
    E_test=0.5*sum((d_test-y_test).^2);
    if(E_test==0)
        E_p=0;
    else E_p=1;
    end
    E_u(train_lt(:,r)+1,1)=E_u(train_lt(:,r)+1,1)+E_p;
    E_d(train_lt(:,r)+1,1)=E_d(train_lt(:,r)+1,1)+1;
end
E_finalt=E_u/E_d;