function  comp=pca_comp_show(mul)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % �ú���������ʾ����PCAѹ�����ͼ��
% % %  param��
% % %     mul--����Ķ���׻��߸߹���ͼ��  
% % %     n--ָ���ö��ٸ����ɷ�
% % % @author��chaolei
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
n=1
[vector ,value,tempMul] = my_pca(mul);
% ʹ����������Ҫ��n�����ɷ�,����ԭ��ԭͼ���С
% PCA��任
% ԭʼ����reshape*������������*������������'
re = tempMul*vector(:,1:n)*vector(:,1:n)';
[r,c,bands] =size(mul);
comp = reshape(re,[r,c,bands]);

str =sprintf('%d%s',n,'�����ɷ�');
figure(1);imshow(comp);title(str);
end
