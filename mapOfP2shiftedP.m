function map=mapOfP2shiftedP(n)
%��������λǰLBPֵ����С��λLBPֵ��map
Key=num2cell(0:2^n-1); 
matVal=zeros(2^n,1);
NoneUniform=0;
for i=0:n-1
    isEven=mod(i,2);
    NoneUniform=NoneUniform+isEven*2^i; %����һ���Ǿ���ģʽ
end
for i=0:2^n-1
    val=zeros(n,1);
    val(n)=i;
    % ����λ���ȡֵ���ϵ�Ӹ�λ����λ�ƶ�
    for j=n-1:-1:1
        p=0; %��λ����
        q=0; %��λ����
        if i<2^j
            q=i;
            p=0;
        else
            q=mod(i,2^j);
            p=(i-q)/2^j;
        end
        val(j)=q*2^(n-j)+p;
    end
    minVal=min(val);
    if isUniformMode(minVal,n) 
        matVal(i+1)=minVal;
    else
        matVal(i+1)=NoneUniform; %�Ǿ���ģʽͳһΪһ��ģʽ
    end
end
Val=num2cell(matVal);
map=containers.Map(Key,Val);
