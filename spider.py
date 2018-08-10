#ʹ��request+bs4
import requests
# from lxml import etree
from bs4 import BeautifulSoup
import re
import os  
import threading
from time import *
# import 
headers = {
   'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64; rv:60.0) Gecko/20100101 Firefox/60.0'
    
    }
# content_xml = content.content.decode('utf-8')
# print(content.text)
# html = etree.HTML(content_xml)
# print(html)
#��ȡ������ҳ��ȫ������
def all_news_info (new_type):#����new_typeΪhealth_
        #��ҳ
    start_url_list = []
    for i in range(1,21):
#         start_urls = 'http://www.yaoyanbaike.com/category/'+str(new_type)+str(i)+'.html'
        if i ==1:     
            start_urls = ('http://www.yaoyanbaike.com/category/'+str(new_type)).replace('_','.html')
            print(start_urls)
            start_url_list.append(start_urls)
        if i>1:
            start_urls = 'http://www.yaoyanbaike.com/category/'+str(new_type)+str(i)+'.html'
            print(start_urls)
#             content = requests.get(start_urls,headers = headers)
#             soup = BeautifulSoup(content.text)
            start_url_list.append(start_urls)
    return start_url_list #�������з���ҳ�棬list��ʽ
#��ȡÿһҳ��ȫ����������
def get_news_url(start_url):
    content = requests.get(start_url,headers = headers)
    soup = BeautifulSoup(content.text,'html.parser')
    #find_all()�������ص���һ���б�
    new_url = soup.find_all('div',class_ = 'list-box m-t-n')
    news = []
    rule = '<a href="(.*?)" title=".*?">.*?</a>'
    new_all = re.findall(rule,str(list(new_url)[0]),re.I)
    for i in new_all:
        #��ȡ����url��������������
        new_url = 'http://www.yaoyanbaike.com'+i
#         print(new_url)
        news.append(new_url)
#     print(news)
    return news #����ÿ�������ȫ��ȫ���������ӣ��б���ʽ
def get_new_info(url,text_num,osdir_name):#��ȡһƪ���µ���ϸ����
#     osdir_name = osdir_name[:-1]#ȥ��������'_'�ַ�
    content = requests.get(url,headers = headers)
    content.encoding='utf-8'#ʹ���������һ���ǿ��Եģ���ֹ��������
    soup = BeautifulSoup(content.text,'html.parser')
    try:
         #�ǵ�ʹ��Beautiful��find_allʹ��class���ҵ�ʱ������class�������࣬������ֱ��ʹ��class,��class��һ���±��_��Ϊclass_
        #find()�����ķ���һ��Ԫ��,����ж�����򷵻ص�һ����ʹ��find().get('��������')������ȡhtmlԪ���еı�ǩ��get('title')��ȡaԪ���е�title���Ե�ֵ
        #find()����һ��Ҫһ���Ĳ��Ҳſ����ҵ�������������һ��ֱ�ӻ�ȡ������Ķ���,Ҫ��-->��--->����ɸ�ֱ�ӵ���
        new_title = soup.find('div',class_ = 'bg-white-only m-b wrapper').find('h1',class_='text-2x m-t-sm')#get_text()ֱ�ӻ�ȡ����ǩ��ȫ���ı�
        if new_title is None:
            pass
        else:
            new_title = new_title.get_text()
        #new_author = soup.find('div',class_ = 'bg-white-only m-b wrapper').find('div',class_='meta text-sm m-t').find('span',class_='m-r-sm text-success').get_text()
        #new_date = soup.find('div',class_ = 'bg-white-only m-b wrapper').find('div',class_='meta text-sm m-t').find('span',class_='text-muted time').get_text()
        new_content = soup.find('div',class_ = 'bg-white-only m-b wrapper').find('article',class_='content-text m-t-sm').get_text()
        article = new_content.replace(u'\n','')
    #     print(new_content.replace(u'\n',''))#ȥ�����������л��з�ʹ��u''��ֹת��
    #     print(new_date[6:])
    #     print(new_author)
    #     print(new_title)
        #������д��txt�ļ���
        with open('e:py/ҥ������/'+str(osdir_name)+'/'+str(text_num)+'.txt',mode='w', encoding='utf-8') as f:
            f.write(article)
            f.close()
            print('д����һƪ����')
    except Exception as E:
        print('��ȡ���󣡣���')
        raise #�鿴�׳����쳣
#         print(E)
        print(url) 
#��ȡ��ҳȫ������   
def get_index_info():
    start_url = 'http://www.yaoyanbaike.com/'
    content = requests.get(start_url,headers = headers,timeout = 500)#��������ʱ��
    soup = BeautifulSoup(content.text,'html.parser')
 
    #�ǵ�ʹ��Beautiful��find_allʹ��class���ҵ�ʱ������class�������࣬������ֱ��ʹ��class,��class��һ���±��_��Ϊclass_
    new_title = soup.find_all('div',class_='media m-n wrapper-sm b-b')#���ص���һ���б�
    
#     new_data = html.xpath('/html/body/section[1]/div/div[1]/div/div[2]/div/div[1]/div/p[1]')
#     print(new_title)
    for span in new_title:
#         print(span)
        new_title = span.find('a',class_='media-left hidden-xs').get('title')
        #find()�����ķ���һ��Ԫ��,����ж�����򷵻ص�һ����ʹ��find().get('��������')������ȡhtmlԪ���еı�ǩ��get('title')��ȡaԪ���е�title���Ե�ֵ
        #��ȡ����url��������������
        new_url = start_url+span.find('a',class_='media-left hidden-xs').get('href')
#         print(new_url)
        #find()����һ��Ҫһ���Ĳ��Ҳſ����ҵ�������������һ��ֱ�ӻ�ȡ������Ķ���,Ҫ��-->��--->����ɸ�ֱ�ӵ���
        new_author = span.find('div',class_ = 'media-body').find('p',class_ = 'm-n l-h-2x').find('span',class_='m-r-xs text-success text-sm').get_text()
#         print(new_author)
        new_date = span.find('div',class_ = 'media-body').find('p',class_ = 'm-n l-h-2x').find('span',class_='time m-r-xs text-success text-sm').get_text()
#         print(new_date)    
    
'''
  ����ƥ��
  rule = '<a class="media-left hidden-xs" href="(.*?)" target="_blank" title="(.*?)">'#��ȡ
    # ne = re.compile(rule)
    data = re.findall(rule,content.text,re.I)
    #��ȡ��ҳÿ�����ŵ���ϸ����
    data_url = []
    for i in range(0,len(data)):
        data_n = start_url+data[i][0]
        data_url.append(data_n)
#     print(data_url)
  
  ''' 

# get_index_url()
#��װ�ɺ������������߳�
def main(type_list):
#     type_list = ['health_','food_','baby_','science_',
#                 'life_','legend_','news_','car_','love_','sexual_'
#                 ]
    # type_list = ['health_']
#     a = 0
    b = 0
    c = 0
    for item in type_list:
    #     all_news_info(item)
        #���ÿ�������µ�20ҳ
        all_list = all_news_info(item)
    #     print(all_list)
        osdir_name = item[:-1]
        os.makedirs('e:py/ҥ������/'+str(osdir_name))
        #ץȡÿһ�������ÿһҳ��ȫ������
        c=0
        for i in all_list:
            b+=1
            news_list = get_news_url(i)#ÿһ�η���ÿһҳ��20ƪ��������
            print(news_list)
            print('��ʼץȡ����ĵ�'+str(b)+'ҳ��')
            #������ȡÿһƪ���µ���ϸ����
            for j in news_list:
                c+=1
                get_new_info(j,c,osdir_name)
            print('��'+str(c+20)+'ƪ����ץȡ��ɣ�')
            print('------------------------')
#     timer = threading.Timer(5*60*60,main)#�����߳�
#     timer.start()#�����߳�

#���������߳��࣬һ����һ��
class One_of_half(threading.Thread):#��һ���̣߳��̳�threading.Thread���࣬Ҫʵ����������__inti__��run()
    def __init__(self,start_list_urls):
        threading.Thread.__init__(self)
        self.start_urls = start_list_urls
    
    #ִ�к����������߳�
    def run(self):
        start_urls = self.start_urls
        main(start_urls)
    
#�̶߳�
class Two_of_half(threading.Thread):
    def __init__(self,start_list_urls):
        threading.Thread.__init__(self)
        self.start_urls = start_list_urls
        
    def run(self):
        start_urls = self.start_urls
        main(start_urls)

if __name__ =='__main__':
    type_list1 = ['health_','food_','baby_','science_',
            'life_'
            ]
    type_list2 = ['legend_','news_','car_','love_','sexual_']
    One = One_of_half(type_list1)
    One.start()
    sleep(0.5)
    Two = Two_of_half(type_list2)
    Two.start()
    
#     timer2 = threading.Timer(1,main)#������һ���߳�
#     timer2.start()#�����߳�
    # news=get_news_url('http://www.yaoyanbaike.com/category/health.html')
# print(news)
# print(news)
# print(news)
# print(data)
# print(new_data)
# get_new_info('http://www.yaoyanbaike.com/a/836nv.html')