
function [vector ,value,tempMul] = my_pca(mul)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % �ú�������ʵ�ֶ����ͼ���PCA�㷨
% % %    param��
% % %          mul--����Ķ����ͼ��Ҳ�����Ƕ�ͨ����
% % %          value--����ֵ�Ӵ�С����
% % %          vector--����������Ӧ����ֵ���У�һ����һ����������
% % %          tempMul--mul��reshape��pixels*bands��ʽ��2ά����
% % % 
% % %  @author��chaolei
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
mul=double(mul)/255;
[r ,c ,bands]=size(mul);
pixels = r*c;
% reshape��pixels*channel
mul = reshape(mul, [pixels,bands]);
tempMul = mul;
% ���ͨ���ľ�ֵ
meanValue =  mean(mul,1);

% ����ȥ���Ļ�
mul = mul - repmat(meanValue,[r*c,1]);
% ��Э�������
correlation = (mul'*mul)/pixels;
%����������������ֵ
[vector ,value] = eig(correlation);
% ����ֵ�����������Ӵ�С����
vector = fliplr(vector);
value = fliplr(value);
value = flipud(value);

end

