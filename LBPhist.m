function h=LBPhist(LBP,n)
%���LBPͳ��ֱ��ͼ
 
%������λǰ��lbpӳ���
map=mapOfP2shiftedP(n);
 
%������λ��lbp��ֱ��ͼbin��ӳ��
v=cell2mat(values(map));
v=unique(v);
key=num2cell(v);
value=num2cell(1:length(v));
map2=containers.Map(key,value);
 
h=zeros(length(v),1);
%ֱ��ͼͳ��
for i=1:size(LBP,1)
    for j=1:size(LBP,2)
        idx=map2(LBP(i,j));
        h(idx)=h(idx)+1;
    end
end
 