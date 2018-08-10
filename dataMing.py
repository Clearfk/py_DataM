#���ŷ���
import numpy as np
from numpy import *
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split#���뽻����֤����
from sklearn.feature_extraction.text import TfidfVectorizer# �����ı�����ֵ������ �����ı������ݽ��д�����Ҫʹ�õ�������
from sklearn.metrics import precision_recall_fscore_support #���������
from sklearn.metrics import accuracy_score
from sklearn import metrics #������������
def get_stop_word():
    stop = open('e:py/����Դ/stop_words.txt','r',encoding = 'utf-8').read()
    stop_word = stop.split(u'\n')
    return stop_word

def get_transform_data(stop_word):
    #��������
    data = load_files('e:/py/����Դ/ҥ������',encoding = 'utf-8')#����load_files������ȡҥ�����Ÿ��ļ������������ļ��е������ı����ݡ�ָ������utf-8

    # # ���ݹ�һ��
    # from sklearn.preprocessing import StandardScaler
    # st = StandardScaler()
    # #�Ի��ֺõ����ݽ��й�һ��
    # new_data= st.fit_transform(data)
    # print(new_data)
#     print(data.data[2])

    #����ѵ���ͺͲ������ݣ�ѵ��70%,����30%
    #������ࣺѵ�����ݡ��������ݡ�ѵ�����ݡ���������
    text_train,text_test,y_train,y_test = train_test_split(data.data,data.target,test_size = 0.3)
    # BOOL�������µ������ռ�ģ�ͣ�ע�⣬�����������õ���transform�ӿ�
    #�������󣬼�ס�������ı��͵����ݵĴ����Ҫʹ�õ�Tfidf����������������������������
    text_vec = TfidfVectorizer(binary=False,decode_error='ignore',stop_words = stop_word)
    #��ȡ����ֵ,ʹ��fit_transform()���������ݹ�һ������׼��
    # x_train = text_vec.fit_transform(text_train)#��text_trainѵ������תΪ����ֵ������ʽ
    # y_train = text_vec.fit_transform(y_train)
    # y_test = text_vec.fit_transform(y_test)
    x_test = text_vec.fit_transform(text_test)#������ѹ������תΪϡ��������ʽ
    # print(np.linalg.eigvals(x_test))
    # print(len(x_test)) #x_test�ĳ���Զ��x�ĳ���ҪС����Ϊʹ��fit_transformѹ����
    # x_test = text_vec.transform(text_test) 

    # fit_transform() = fit()+transform(),��ʹ����fit_transform()
    x = text_vec.transform(data.data)#����������תΪ�������� ,Ҫ��ʹ��fit_transform�����Ļ��������ʹ��fit(x,y)�ķ��������ת�������Σ�������
    #����ʹ��predict��ʱ�򱨴���Ϊת���������ˣ�����ֵ���С�ܶ�
    y = data.target#��ȡ���ݵ�����ǩ
    # print(x_train.toarray())
    # print(text_vec.get_feature_names())
    # print(data.data)
    # print(data.target)
    return x,y,y_test,x_test

#��Ҷ˹����
def bayes(x,y,y_test,x_test):
# ����bayes����
    from sklearn.naive_bayes import BernoulliNB #����bayesģ��
#     from sklearn import metrics #�������
    bayes_model = BernoulliNB()
    bayes_model.fit(x,y)
    # print(model1)
    #����
    expected = y_test#��������
    # print(x_test.shape[0:])
    # print(x_test.shape)
    # print()
    # print(model1.n_features)
    # a = list(x_test.shape)[0]*list(x_test.shape)[1]
    # c,b = x_test.shape
    # print(c)
    # print(x_test.shape[1])
    x_shape = x_test
    predicted = bayes_model.predict(x_shape)#Ԥ������
    #������
    ture_rate = accuracy_score(y_test,predicted)
    print('ģ����ȷ��Ϊ: %f' % ture_rate)
    labels = set(y)
    print(metrics.classification_report(expected,predicted)) #���������
    matrix_info = metrics.confusion_matrix(expected,predicted) #��ȡ��������
    print(matrix_info)#�����������
    print('��'+str(len(list(labels)))+'��')
    p,r,f1,s = precision_recall_fscore_support(expected,predicted)
    return matrix_info,s,labels

#knn����
def knn(x,y,y_test,x_test):
#     from sklearn import metrics
    from sklearn.neighbors import KNeighborsClassifier
    knn_model = KNeighborsClassifier(n_neighbors = 3)#�趨3���ٽ���Ϊһ����
    knn_model.fit(x,y)
    expected = y_test
#     x_shape = x_test
    predicted = knn_model.predict(x_test)
    ture_rate = accuracy_score(y_test,predicted)
    print('ģ����ȷ��Ϊ: %f' % ture_rate)
    labels = set(y)
    print(metrics.classification_report(expected,predicted,digits=5))
    matrix_info = metrics.confusion_matrix(expected,predicted)
    print(matrix_info)
    print('��'+str(len(list(labels)))+'��')
    p,r,f1,s = precision_recall_fscore_support(expected,predicted)
    return matrix_info,s,labels
    
    
#����������
def decis_tree(x,y,y_test,x_test):
#     from sklearn import metrics
    from sklearn import tree
    tree_model = tree.DecisionTreeClassifier(criterion = 'gini')
    tree_model = tree.DecisionTreeClassifier(criterion = 'entropy') #Ϊģ������������ʼ����
    tree_model.fit(x,y)
    expected = y_test
    predicted = tree_model.predict(x_test)
    ture_rate = accuracy_score(y_test,predicted)
    print('ģ����ȷ��Ϊ: %f' % ture_rate)
    labels = set(y)
    print(metrics.classification_report(expected,predicted,digits=5))
    matrix_info = metrics.confusion_matrix(expected,predicted)
    print(matrix_info)
    print('��'+str(len(list(labels)))+'��')
    p,r,f1,s = precision_recall_fscore_support(expected,predicted)
    return matrix_info,s,labels
    
#svm֧������������
# SVM�ȿ�������������࣬����SVC���ֿ�������Ԥ�⣬���߳�Ϊ�ع飬����SVR��sklearn�е�svmģ����Ҳ������SVR�ࡣ

def svm(x,y,y_test,x_test):
#     from sklearn import metrics
    from sklearn import svm
    from sklearn.multiclass import OneVsRestClassifier
    svm_model = svm.SVC(kernel = 'rbf',degree = 2,gamma = 1.7)
    svm_model.fit(x,y)
    expected = y_test
    predicted = svm_model.predict(x_test)
    ture_rate = accuracy_score(y_test,predicted)
    print('ģ����ȷ��Ϊ: %f' % ture_rate)
    print(metrics.classification_report(expected,predicted,digits=5))
    labels = set(y)
    matrix_info = metrics.confusion_matrix(expected,predicted)
    print(matrix_info)
#     print(matrix_info[0][0]) #�����һ���Ԥ����ȷ����
    print('��'+str(len(list(labels)))+'��')
    p,r,f1,s = precision_recall_fscore_support(expected,predicted)
#     print(s)
    return svm_model,matrix_info,s,labels #����ģ�Ͷ���
#     feature_weight = svm_model.feature_importances_ #��ȡģ�����ݵ�����������Ȩ��
#     print(feature_weight)

def draw_model(model_name,matrix_info,s,labels,kind,savepath):
    import pandas as pd
    import matplotlib.pyplot as plt
#     print(matrix_info)
    
    ture_data = list(s)
    predict = []
    for i in range(0,len(labels)):
        for j in range(0,len(labels)):
            if i ==j:
                predict.append(matrix_info[i][j])
    print(predict)
    print(ture_data)
    d={
        'Predict':predict,
        'Ture':ture_data
    }
    df = pd.DataFrame(d,
        # ����������ʾ��˳��
        columns=['Ture','Predict'],
        index=list(labels))
    plt.figure()
    df.plot(kind =kind,alpha = 0.8,rot = 0)
#     df.plot()
#     df.plot(kind = 'line',alpha = 0.5,rot = 0)#������ͼ
#     x2.plot(kind = 'bar',color = 'red',stacked=True)
    plt.legend(df.columns,loc = 'upper right',frameon = False)  #���ñ�ǩע�����ұ��м�λ����ʾ
    plt.title(model_name+'ҥ�����ŷ���ģ��')
    plt.ylabel('������')
    plt.xlabel('���')
    plt.yticks([y for y in range(0,180,10)])#����y��̶�
    for i in range(0,len(list(labels))):
    #�����Ӽ�������ʹ��text����
    #plt.text()
   #  ��һ��������x�����꣬�ڶ���������y�����꣬������������Ҫ��ʽ�����ݣ� alpha ���������͸���ȣ�family �������壬 size ��������Ĵ�С
   # style ��������ķ��
   # wight ����Ĵ�ϸ
   # bbox ��������ӿ�alpha ���ÿ����͸���ȣ� facecolor ���ÿ������ɫ
        plt.text(i-0.18,df.get("Ture")[i],'%.0f'%df.get("Ture")[i], ha='center', va='bottom',alpha = 0.7)
        plt.text(i+0.3,df.get("Predict")[i],'%.0f'%df.get("Predict")[i], ha='center', va='bottom',alpha=0.7)
    plt.savefig(savepath,dpi = 150)
    plt.show()


#svm���ROC��������
#��ROC����
# ��1��ROC������ʵ���������������չʾĳ���ж�ԭ��Ч���ò��һ��ͼ�Σ�����ͨ��AUC[0,1]��������С��
# ��2�����������ֵ�󣬿���ͨ�������ȡ�����ȡ���ȷ���������жϵľ���Ч����

# ���������1-����ȣ������������ȡ���ô�ͻ��γ�1�����������ߡ�
# ������ߺ�45�ȵ�ֱ�߻��γ�һ�����������(area under ROC)�����AUC��AUCԽ��˵���жϵ�Ч��Խ�á�
# ��ͼ��ʾ��AUCΪ0.9758��˵���ж�Ч�������ˣ�
# ���ǣ�ʵ�ʹ����У�һ��AUC��0.7-0.9��Χ�ڵıȽϳ���������0.9�����ڷ�ë����ˡ�
# ��Ȼ���������Լ��ķ������������Ļ�����������רҵ��ͳ��ʦŶ�� 
# def svm_and_ROC(x,y,y_test,x_test,savepath):
#     from sklearn import metrics
#     from sklearn.metrics import roc_curve,auc
#     import matplotlib.pyplot as plt
#     from sklearn import svm
#     from sklearn.multiclass import OneVsRestClassifier
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import label_binarize
#     # ROCͼ�Ļ滭
#     classes = [ i for i in range(len(list(set(y))))]
# #     classes = [0,1,2,3,4,5,6,7,8,9]
#     y = label_binarize(y,classes = classes)
#     n_classes = y.shape[1]
#     n_sample,n_features = x_test.shape#��ȡ�������ݵ�shapeֵ����ֵ
#     #y��ֵ�󣬶��������·���
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#     svm_model2 = OneVsRestClassifier(svm.SVC(kernel = 'rbf',degree = 2,gamma = 1.7)) #��������Ϊsvm_model2��ֹ�������svm()����������svm_model����
#     y_score = svm_model2.fit(x,y).decision_function(X_test)
#     #����ÿ������rocֵ
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     #�������
#     for i in range(n_classes):
#         fpr[i],tpr[i],_ = roc_curve(y_test[:,i],y_score[:,i])
#         roc_auc[i] = auc(fpr[i],tpr[i])
    
#     fpr['micro'],tpr['micro'],_ = roc_curve(y_test.ravel(),y_score.ravel())
#     roc_auc['micro'] = auc(fpr['micro'],tpr['micro'])
#     #��ͼ
#     plt.figure()
#     lw = 2
#     plt.plot(fpr[2],tpr[2],color = 'b',lw = lw,label = 'ROC curve(area = %0.2f)' % roc_auc[2])
#     plt.plot([0,1],[0,1],color = 'red',lw = lw,linestyle='--')
#     plt.xlim([0.0,1.0])
#     plt.ylim([0.0,1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC')
#     plt.legend(loc = 'lower right')
#     plt.savefig(savepath,dpi = 150)
#     plt.show()
    
#������װ��ͼ���� ��ֻ�ܻ�svm��ģ��ROC  
def draw_ROC(x,y,y_test,x_test,model_type,savepath):
#     from sklearn import metrics
    from sklearn.metrics import roc_curve,auc
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    # ROCͼ�Ļ滭
    classes = [ i for i in range(len(list(set(y))))]
#     classes = [0,1,2,3,4,5,6,7,8,9]
    y = label_binarize(y,classes = classes)
    n_classes = y.shape[1]
    n_sample,n_features = x_test.shape#��ȡ�������ݵ�shapeֵ����ֵ
    #y��ֵ�󣬶��������·��࣬��������ȡ30%
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model2 = OneVsRestClassifier(model_type) #��������Ϊsvm_model2��ֹ�������svm()����������svm_model����
    y_score = model2.fit(x,y).decision_function(X_test)
    #����ÿ������rocֵ
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    #�������
    for i in range(n_classes):
        fpr[i],tpr[i],_ = roc_curve(y_test[:,i],y_score[:,i])
        roc_auc[i] = auc(fpr[i],tpr[i])
    #����auc
    fpr['micro'],tpr['micro'],_ = roc_curve(y_test.ravel(),y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'],tpr['micro'])
    #��ͼ
    plt.figure()
    lw = 2
    plt.plot(fpr[2],tpr[2],color = 'b',lw = lw,label = 'ROC curve(area = %0.2f)' % roc_auc[2])
    plt.plot([0,1],[0,1],color = 'red',lw = lw,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM֧������--ROC����')
    plt.legend(loc = 'lower right')
    plt.savefig(savepath,dpi = 150)
    plt.show()
    
    
# ������޼ලѧϰ�ģ���������û�н���ѵ���Ϳ���ֱ�Ӷ����ݼ�x_test���з�����㷨���������Ҫfit()�Ķ��Ǽලѧϰ���㷨
def k_means(x_test,y_test):
    #����sklearn�е���kmeansģ��
    from sklearn.cluster import Birch
    from sklearn.cluster import KMeans  #����kmeansģ��
    #����ģ�Ͷ��� ���� n_clusters = 'Ҫ�ֳɵ������'   , n_jobs = 2,ģ�Ϳ����ı�ʾ�߳���Ϊ2��max_iter = 'ģ��ѭ����֤�Ĵ���'
    kms_model = KMeans(n_clusters = 3,n_jobs = 3,max_iter = 200)
    predicted = kms_model.fit_predict(x_test)
    print(x_test.shape)
    print(len(predicted))
    expected = y_test
    print(metrics.classification_report(expected,predicted))
    print(metrics.confusion_matrix(expected,predicted))
    print('��'+str(len(predicted))+'��') 

    
def single_transform(x,x_test1):
    from sklearn.feature_extraction.text import CountVectorizer
    from collections import Counter
    import numpy as np
    from scipy.sparse import coo_matrix
    #��ϡ�������ʽ��x_test1תΪһ�����飬ʹ��toarray()����
    new_array = x_test1.toarray()
    # print(new_array)
    #����x_test1Ҫת�ɵ�ά������
    b = np.zeros((1,x.shape[1]))
    # �������������ά����
#     print(b.shape[1])
    # e = np.array([[1,2,8]])
    # t = np.array([[0,0,0,0]])
    # for i in range(0,len(e[0])):
    #     print(e[0][i])
    #     t[0][i] = e[0][i]
    # print(t)
    #����ѭ������new_array�е���ֵתΪά��Ϊ10000���������b
    for j in range(0,len(new_array[0])):
        b[0][j] = new_array[0][j]
    #���תΪ10000ά�ȵ�new_array����
#     print(b)
    #�������ά��
#     print(b.shape[1])
    final_data = coo_matrix(b) #����coo_matrix()��һ������bתΪϡ�����
#     print(final_data.shape)
    #���ά�ȱ�Ϊ10000ά�ȵ�x_test1��������
    print('����ά�Ⱥ��ϡ������ά��Ϊ: %s' %final_data.shape[1])
    return final_data



#�����������ݣ�ʵ���С�������
def singel_predict_svm(x,y,stop_word,fname):
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import CountVectorizer
    import io
    import numpy as np
    from sklearn import metrics
    from sklearn import svm
    from sklearn.multiclass import OneVsRestClassifier
    #     print(X_test)
#     from sklearn.preprocessing import  
    #��ȡ��������
#     data2 = load_files('e:/py/health',encoding = 'utf-8')
#     print(data2)
    file_name = fname
    data1 = open(file_name,'r',encoding = 'utf-8').read()
#     print([data1])
    data2 = np.array([data1])
    #�ı�����ת����ȡ����ֵ����һ
    text_vec = CountVectorizer(stop_words = stop_word)
    x_test1 =text_vec.fit_transform(data2)#data2����Ϊnp.array�Ķ�ά����
#     print(x_test1.shape)  
#   #�ı�����ת����ȡ����ֵ������
#     text_vec = TfidfVectorizer(binary=False,decode_error='ignore',stop_words = stop_word)
    #��ȡ����ֵ,ʹ��fit_transform()���������ݹ�һ������׼��
#     x_test1 = text_vec.fit_transform(data2)#������ѹ������תΪϡ��������ʽ,data2����Ϊnp.array��ά����
#     print(x_test.shape)
    
    svm_model = svm.SVC(kernel = 'rbf',degree = 2,gamma = 1.7)
    svm_model.fit(x,y) 
    print('�����������ݵ�ά��Ϊ��%s' % x_test1.shape[1]) #�������ݵ�ά����
    print('ѵ�����ݵ�ά��Ϊ��%s ' %x.shape[1])#ѵ�����ݵ�ά������
    #�������ݵ�����ά����һ��Ҫ��ѵ�����ݵ�����ά������Ȳſ���
    #���������ݵ�ά����תΪ��ѵ�����ݵ�ά������ȵ�ά��
    x_test2 = single_transform(x,x_test1)
    predicted = svm_model.predict(x_test2)

#     print(metrics.classification_report(expected,predicted))
#     labels = set(y)
#     print(metrics.confusion_matrix(expected,predicted))
#     print('��'+str(len(list(labels)))+'��') 
#     print(predicted)
#     print('����Ӧ����Ϊ��')
#     print('0:baby,1:car,2:food,3:health,4:legend,5:life,6:love,7:news,8:science,9:sexual')
#     print('Ԥ�����������ǣ�%s' % predicted[0])
    return predicted[0]
def judge_type(num):
    a = {
        0:'baby',1:'car',2:'food',3:'health',4:'legend',5:'life',6:'love',7:'news',8:'science',9:'sexual'
    }
    for i in range(0,len(list(a.keys()))):
        if num == list(a.keys())[i]:
#             print(list(a.keys())[i])
            print('Ԥ������ŵ����Ϊ: %s' %list(a.values())[i])
    
    
if __name__=='__main__':
    stop_word = get_stop_word()
    x,y,y_test,x_test= get_transform_data(stop_word)
    print('------------bayesģ�ͷ��������----------------')
    g,h,l= bayes(x,y,y_test,x_test)
    draw_model('bayes',g,h,l,'bar','e:py/����Դ/bayes_pic.png')
    draw_model('bayes',g,h,l,'line','e:py/����Դ/bayes_line_pic.png')
    print('-------------������ģ�ͷ��������-------------------')
    a,b,c = decis_tree(x,y,y_test,x_test)
    draw_model('������',a,b,c,'bar','e:py/����Դ/tree_pic.png')
    draw_model('������',a,b,c,'line','e:py/����Դ/tree_line_pic.png')
    print('--------------knn����ģ�ͽ��------------------')
    d,e,f = knn(x,y,y_test,x_test)
    draw_model('knn',d,e,f,'bar','e:py/����Դ/knn_pic.png')
    draw_model('knn',d,e,f,'line','e:py/����Դ/knn_line_pic.png')
    print('------------------svm֧������������ģ�ͽ��------------')
    svm,matrix_info,s,labels = svm(x,y,y_test,x_test)
    draw_model('svm֧������',matrix_info,s,labels,'bar','e:py/����Դ/svm_pic.png')
    draw_model('svm֧������',matrix_info,s,labels,'line','e:py/����Դ/svm_line_pic.png')
    print('------------------svm֧������������ģ��ROC����------------')
    draw_ROC(x,y,y_test,x_test,svm,'e:py/����Դ/svm_ROC.png')
#     svm_and_ROC(x,y,y_test,x_test,'e:py/����Դ/ROC.png')
#     ROC(y,y_score,'e:py/����Դ/ROC.png')
#     print('--------------�Ǽලѧϰ�����㷨K-Means------------')
#     k_means(x_test,y_test)
#     print(x.shape)
#     print(x_test.shape[1])
#     num = singel_predict_svm(x,y,stop_word,'e:/py/����Դ/ҥ������/food/1.txt')
    num = singel_predict_svm(x,y,stop_word,'e:/py/����Դ/ҥ������/health/1.txt')
    print('���Ϊ��%s' %num)
    judge_type(num)
