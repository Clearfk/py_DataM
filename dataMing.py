#新闻分类
import numpy as np
from numpy import *
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split#引入交叉验证函数
from sklearn.feature_extraction.text import TfidfVectorizer# 导入文本特征值得求解库 ，对文本的数据进行处理都是要使用到这个库的
from sklearn.metrics import precision_recall_fscore_support #结果分析库
from sklearn.metrics import accuracy_score
from sklearn import metrics #分类结果分析库
def get_stop_word():
    stop = open('e:py/数据源/stop_words.txt','r',encoding = 'utf-8').read()
    stop_word = stop.split(u'\n')
    return stop_word

def get_transform_data(stop_word):
    #导入数据
    data = load_files('e:/py/数据源/谣言新闻',encoding = 'utf-8')#利用load_files方法读取谣言新闻父文件下面所有子文件夹的所有文本内容。指定编码utf-8

    # # 数据归一化
    # from sklearn.preprocessing import StandardScaler
    # st = StandardScaler()
    # #对划分好的数据进行归一化
    # new_data= st.fit_transform(data)
    # print(new_data)
#     print(data.data[2])

    #划分训练和和测试数据，训练70%,测试30%
    #交叉分类：训练数据、测试数据、训练数据、测试数据
    text_train,text_test,y_train,y_test = train_test_split(data.data,data.target,test_size = 0.3)
    # BOOL型特征下的向量空间模型，注意，测试样本调用的是transform接口
    #创建对象，记住！对于文本型的数据的处理就要使用到Tfidf这个库来处理，其他处理都不够这个好
    text_vec = TfidfVectorizer(binary=False,decode_error='ignore',stop_words = stop_word)
    #获取特征值,使用fit_transform()方法将数据归一化，标准化
    # x_train = text_vec.fit_transform(text_train)#将text_train训练数据转为特征值数组形式
    # y_train = text_vec.fit_transform(y_train)
    # y_test = text_vec.fit_transform(y_test)
    x_test = text_vec.fit_transform(text_test)#将数据压缩，并转为稀疏矩阵的形式
    # print(np.linalg.eigvals(x_test))
    # print(len(x_test)) #x_test的长度远比x的长度要小，因为使用fit_transform压缩了
    # x_test = text_vec.transform(text_test) 

    # fit_transform() = fit()+transform(),故使用了fit_transform()
    x = text_vec.transform(data.data)#将所有数据转为特征向量 ,要是使用fit_transform方法的话，如果在使用fit(x,y)的方法后就是转化了两次，这样就
    #会在使用predict的时候报错，因为转化了两次了，特征值会变小很多
    y = data.target#获取数据的类别标签
    # print(x_train.toarray())
    # print(text_vec.get_feature_names())
    # print(data.data)
    # print(data.target)
    return x,y,y_test,x_test

#贝叶斯分类
def bayes(x,y,y_test,x_test):
# 创建bayes对象
    from sklearn.naive_bayes import BernoulliNB #导入bayes模型
#     from sklearn import metrics #结果报告
    bayes_model = BernoulliNB()
    bayes_model.fit(x,y)
    # print(model1)
    #测试
    expected = y_test#测试数据
    # print(x_test.shape[0:])
    # print(x_test.shape)
    # print()
    # print(model1.n_features)
    # a = list(x_test.shape)[0]*list(x_test.shape)[1]
    # c,b = x_test.shape
    # print(c)
    # print(x_test.shape[1])
    x_shape = x_test
    predicted = bayes_model.predict(x_shape)#预测数据
    #输出结果
    ture_rate = accuracy_score(y_test,predicted)
    print('模型正确率为: %f' % ture_rate)
    labels = set(y)
    print(metrics.classification_report(expected,predicted)) #输出分类结果
    matrix_info = metrics.confusion_matrix(expected,predicted) #获取混淆矩阵
    print(matrix_info)#输出混淆矩阵
    print('共'+str(len(list(labels)))+'类')
    p,r,f1,s = precision_recall_fscore_support(expected,predicted)
    return matrix_info,s,labels

#knn分类
def knn(x,y,y_test,x_test):
#     from sklearn import metrics
    from sklearn.neighbors import KNeighborsClassifier
    knn_model = KNeighborsClassifier(n_neighbors = 3)#设定3个临近点为一个类
    knn_model.fit(x,y)
    expected = y_test
#     x_shape = x_test
    predicted = knn_model.predict(x_test)
    ture_rate = accuracy_score(y_test,predicted)
    print('模型正确率为: %f' % ture_rate)
    labels = set(y)
    print(metrics.classification_report(expected,predicted,digits=5))
    matrix_info = metrics.confusion_matrix(expected,predicted)
    print(matrix_info)
    print('共'+str(len(list(labels)))+'类')
    p,r,f1,s = precision_recall_fscore_support(expected,predicted)
    return matrix_info,s,labels
    
    
#决策树分类
def decis_tree(x,y,y_test,x_test):
#     from sklearn import metrics
    from sklearn import tree
    tree_model = tree.DecisionTreeClassifier(criterion = 'gini')
    tree_model = tree.DecisionTreeClassifier(criterion = 'entropy') #为模型设置两个初始参数
    tree_model.fit(x,y)
    expected = y_test
    predicted = tree_model.predict(x_test)
    ture_rate = accuracy_score(y_test,predicted)
    print('模型正确率为: %f' % ture_rate)
    labels = set(y)
    print(metrics.classification_report(expected,predicted,digits=5))
    matrix_info = metrics.confusion_matrix(expected,predicted)
    print(matrix_info)
    print('共'+str(len(list(labels)))+'类')
    p,r,f1,s = precision_recall_fscore_support(expected,predicted)
    return matrix_info,s,labels
    
#svm支持向量机分类
# SVM既可以用来多类分类，就是SVC；又可以用来预测，或者成为回归，就是SVR。sklearn中的svm模块中也集成了SVR类。

def svm(x,y,y_test,x_test):
#     from sklearn import metrics
    from sklearn import svm
    from sklearn.multiclass import OneVsRestClassifier
    svm_model = svm.SVC(kernel = 'rbf',degree = 2,gamma = 1.7)
    svm_model.fit(x,y)
    expected = y_test
    predicted = svm_model.predict(x_test)
    ture_rate = accuracy_score(y_test,predicted)
    print('模型正确率为: %f' % ture_rate)
    print(metrics.classification_report(expected,predicted,digits=5))
    labels = set(y)
    matrix_info = metrics.confusion_matrix(expected,predicted)
    print(matrix_info)
#     print(matrix_info[0][0]) #输出第一类的预测正确个数
    print('共'+str(len(list(labels)))+'类')
    p,r,f1,s = precision_recall_fscore_support(expected,predicted)
#     print(s)
    return svm_model,matrix_info,s,labels #返回模型对象
#     feature_weight = svm_model.feature_importances_ #获取模型数据的特征向量的权重
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
        # 定义数据显示的顺序
        columns=['Ture','Predict'],
        index=list(labels))
    plt.figure()
    df.plot(kind =kind,alpha = 0.8,rot = 0)
#     df.plot()
#     df.plot(kind = 'line',alpha = 0.5,rot = 0)#画折线图
#     x2.plot(kind = 'bar',color = 'red',stacked=True)
    plt.legend(df.columns,loc = 'upper right',frameon = False)  #设置标签注释在右边中间位置显示
    plt.title(model_name+'谣言新闻分类模型')
    plt.ylabel('测试数')
    plt.xlabel('类别')
    plt.yticks([y for y in range(0,180,10)])#设置y轴刻度
    for i in range(0,len(list(labels))):
    #给柱子加文字是使用text方法
    #plt.text()
   #  第一个参数是x轴坐标，第二个参数是y轴坐标，第三个参数是要显式的内容， alpha 设置字体的透明度，family 设置字体， size 设置字体的大小
   # style 设置字体的风格
   # wight 字体的粗细
   # bbox 给字体添加框，alpha 设置框体的透明度， facecolor 设置框体的颜色
        plt.text(i-0.18,df.get("Ture")[i],'%.0f'%df.get("Ture")[i], ha='center', va='bottom',alpha = 0.7)
        plt.text(i+0.3,df.get("Predict")[i],'%.0f'%df.get("Predict")[i], ha='center', va='bottom',alpha=0.7)
    plt.savefig(savepath,dpi = 150)
    plt.show()


#svm结合ROC曲线例子
#画ROC曲线
# （1）ROC曲线其实是诊断试验中用于展示某个判断原则效果好差的一种图形，可以通过AUC[0,1]来衡量大小。
# （2）给定最佳阈值后，可以通过灵敏度、特异度、正确率来评价判断的具体效果。

# 如果横轴是1-特异度，纵轴是灵敏度。那么就会形成1个弯曲的曲线。
# 这个曲线和45度的直线会形成一个曲线下面积(area under ROC)，简称AUC。AUC越大，说明判断的效果越好。
# 如图所示，AUC为0.9758，说明判断效果优秀了！
# 但是，实际工作中，一般AUC在0.7-0.9范围内的比较常见。超过0.9的属于凤毛麟角了。
# 当然，如果你对自己的分析结果不满意的话，可以求助专业的统计师哦。 
# def svm_and_ROC(x,y,y_test,x_test,savepath):
#     from sklearn import metrics
#     from sklearn.metrics import roc_curve,auc
#     import matplotlib.pyplot as plt
#     from sklearn import svm
#     from sklearn.multiclass import OneVsRestClassifier
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import label_binarize
#     # ROC图的绘画
#     classes = [ i for i in range(len(list(set(y))))]
# #     classes = [0,1,2,3,4,5,6,7,8,9]
#     y = label_binarize(y,classes = classes)
#     n_classes = y.shape[1]
#     n_sample,n_features = x_test.shape#获取测试数据的shape值行列值
#     #y二值后，对数据重新分类
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#     svm_model2 = OneVsRestClassifier(svm.SVC(kernel = 'rbf',degree = 2,gamma = 1.7)) #这里命名为svm_model2防止与上面的svm()函数依赖的svm_model重名
#     y_score = svm_model2.fit(x,y).decision_function(X_test)
#     #计算每个类别的roc值
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     #遍历类别
#     for i in range(n_classes):
#         fpr[i],tpr[i],_ = roc_curve(y_test[:,i],y_score[:,i])
#         roc_auc[i] = auc(fpr[i],tpr[i])
    
#     fpr['micro'],tpr['micro'],_ = roc_curve(y_test.ravel(),y_score.ravel())
#     roc_auc['micro'] = auc(fpr['micro'],tpr['micro'])
#     #画图
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
    
#函数封装画图方法 ，只能画svm的模型ROC  
def draw_ROC(x,y,y_test,x_test,model_type,savepath):
#     from sklearn import metrics
    from sklearn.metrics import roc_curve,auc
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    # ROC图的绘画
    classes = [ i for i in range(len(list(set(y))))]
#     classes = [0,1,2,3,4,5,6,7,8,9]
    y = label_binarize(y,classes = classes)
    n_classes = y.shape[1]
    n_sample,n_features = x_test.shape#获取测试数据的shape值行列值
    #y二值后，对数据重新分类，测试数据取30%
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model2 = OneVsRestClassifier(model_type) #这里命名为svm_model2防止与上面的svm()函数依赖的svm_model重名
    y_score = model2.fit(x,y).decision_function(X_test)
    #计算每个类别的roc值
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    #遍历类别
    for i in range(n_classes):
        fpr[i],tpr[i],_ = roc_curve(y_test[:,i],y_score[:,i])
        roc_auc[i] = auc(fpr[i],tpr[i])
    #计算auc
    fpr['micro'],tpr['micro'],_ = roc_curve(y_test.ravel(),y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'],tpr['micro'])
    #画图
    plt.figure()
    lw = 2
    plt.plot(fpr[2],tpr[2],color = 'b',lw = lw,label = 'ROC curve(area = %0.2f)' % roc_auc[2])
    plt.plot([0,1],[0,1],color = 'red',lw = lw,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM支持向量--ROC曲线')
    plt.legend(loc = 'lower right')
    plt.savefig(savepath,dpi = 150)
    plt.show()
    
    
# 这个是无监督学习的，就是事先没有进行训练就可以直接对数据集x_test进行分类的算法，上面的需要fit()的都是监督学习的算法
def k_means(x_test,y_test):
    #导入sklearn中导入kmeans模型
    from sklearn.cluster import Birch
    from sklearn.cluster import KMeans  #导入kmeans模型
    #创建模型对象 参数 n_clusters = '要分成的类别数'   , n_jobs = 2,模型开启的表示线程数为2，max_iter = '模型循环验证的次数'
    kms_model = KMeans(n_clusters = 3,n_jobs = 3,max_iter = 200)
    predicted = kms_model.fit_predict(x_test)
    print(x_test.shape)
    print(len(predicted))
    expected = y_test
    print(metrics.classification_report(expected,predicted))
    print(metrics.confusion_matrix(expected,predicted))
    print('共'+str(len(predicted))+'类') 

    
def single_transform(x,x_test1):
    from sklearn.feature_extraction.text import CountVectorizer
    from collections import Counter
    import numpy as np
    from scipy.sparse import coo_matrix
    #将稀疏矩阵形式的x_test1转为一般数组，使用toarray()方法
    new_array = x_test1.toarray()
    # print(new_array)
    #构造x_test1要转成的维度数组
    b = np.zeros((1,x.shape[1]))
    # 输出构造的数组的维度数
#     print(b.shape[1])
    # e = np.array([[1,2,8]])
    # t = np.array([[0,0,0,0]])
    # for i in range(0,len(e[0])):
    #     print(e[0][i])
    #     t[0][i] = e[0][i]
    # print(t)
    #利用循环，把new_array中的数值转为维度为10000的数组矩阵b
    for j in range(0,len(new_array[0])):
        b[0][j] = new_array[0][j]
    #输出转为10000维度的new_array数组
#     print(b)
    #新数组的维度
#     print(b.shape[1])
    final_data = coo_matrix(b) #利用coo_matrix()将一般数组b转为稀疏矩阵
#     print(final_data.shape)
    #输出维度变为10000维度的x_test1的特征数
    print('增加维度后的稀疏矩阵的维度为: %s' %final_data.shape[1])
    return final_data



#单个测试数据，实现中。。。。
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
    #读取单个数据
#     data2 = load_files('e:/py/health',encoding = 'utf-8')
#     print(data2)
    file_name = fname
    data1 = open(file_name,'r',encoding = 'utf-8').read()
#     print([data1])
    data2 = np.array([data1])
    #文本数据转换提取特征值方法一
    text_vec = CountVectorizer(stop_words = stop_word)
    x_test1 =text_vec.fit_transform(data2)#data2必须为np.array的二维数组
#     print(x_test1.shape)  
#   #文本数据转换提取特征值方法二
#     text_vec = TfidfVectorizer(binary=False,decode_error='ignore',stop_words = stop_word)
    #获取特征值,使用fit_transform()方法将数据归一化，标准化
#     x_test1 = text_vec.fit_transform(data2)#将数据压缩，并转为稀疏矩阵的形式,data2必须为np.array二维数组
#     print(x_test.shape)
    
    svm_model = svm.SVC(kernel = 'rbf',degree = 2,gamma = 1.7)
    svm_model.fit(x,y) 
    print('单个测试数据的维度为：%s' % x_test1.shape[1]) #测试数据的维度数
    print('训练数据的维度为：%s ' %x.shape[1])#训练数据的维度数，
    #测试数据的特征维度数一定要与训练数据的特征维度数相等才可以
    #将测试数据的维度数转为与训练数据的维度数相等的维度
    x_test2 = single_transform(x,x_test1)
    predicted = svm_model.predict(x_test2)

#     print(metrics.classification_report(expected,predicted))
#     labels = set(y)
#     print(metrics.confusion_matrix(expected,predicted))
#     print('共'+str(len(list(labels)))+'类') 
#     print(predicted)
#     print('类别对应数字为：')
#     print('0:baby,1:car,2:food,3:health,4:legend,5:life,6:love,7:news,8:science,9:sexual')
#     print('预测该新闻类别是：%s' % predicted[0])
    return predicted[0]
def judge_type(num):
    a = {
        0:'baby',1:'car',2:'food',3:'health',4:'legend',5:'life',6:'love',7:'news',8:'science',9:'sexual'
    }
    for i in range(0,len(list(a.keys()))):
        if num == list(a.keys())[i]:
#             print(list(a.keys())[i])
            print('预测该新闻的类别为: %s' %list(a.values())[i])
    
    
if __name__=='__main__':
    stop_word = get_stop_word()
    x,y,y_test,x_test= get_transform_data(stop_word)
    print('------------bayes模型分类器结果----------------')
    g,h,l= bayes(x,y,y_test,x_test)
    draw_model('bayes',g,h,l,'bar','e:py/数据源/bayes_pic.png')
    draw_model('bayes',g,h,l,'line','e:py/数据源/bayes_line_pic.png')
    print('-------------决策树模型分类器结果-------------------')
    a,b,c = decis_tree(x,y,y_test,x_test)
    draw_model('决策树',a,b,c,'bar','e:py/数据源/tree_pic.png')
    draw_model('决策树',a,b,c,'line','e:py/数据源/tree_line_pic.png')
    print('--------------knn分类模型结果------------------')
    d,e,f = knn(x,y,y_test,x_test)
    draw_model('knn',d,e,f,'bar','e:py/数据源/knn_pic.png')
    draw_model('knn',d,e,f,'line','e:py/数据源/knn_line_pic.png')
    print('------------------svm支持向量机分类模型结果------------')
    svm,matrix_info,s,labels = svm(x,y,y_test,x_test)
    draw_model('svm支持向量',matrix_info,s,labels,'bar','e:py/数据源/svm_pic.png')
    draw_model('svm支持向量',matrix_info,s,labels,'line','e:py/数据源/svm_line_pic.png')
    print('------------------svm支持向量机分类模型ROC曲线------------')
    draw_ROC(x,y,y_test,x_test,svm,'e:py/数据源/svm_ROC.png')
#     svm_and_ROC(x,y,y_test,x_test,'e:py/数据源/ROC.png')
#     ROC(y,y_score,'e:py/数据源/ROC.png')
#     print('--------------非监督学习聚类算法K-Means------------')
#     k_means(x_test,y_test)
#     print(x.shape)
#     print(x_test.shape[1])
#     num = singel_predict_svm(x,y,stop_word,'e:/py/数据源/谣言新闻/food/1.txt')
    num = singel_predict_svm(x,y,stop_word,'e:/py/数据源/谣言新闻/health/1.txt')
    print('类别为：%s' %num)
    judge_type(num)
