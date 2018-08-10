#使用request+bs4
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
#获取所有主页面全部链接
def all_news_info (new_type):#参数new_type为health_
        #翻页
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
    return start_url_list #返回所有分类页面，list格式
#获取每一页的全部文章链接
def get_news_url(start_url):
    content = requests.get(start_url,headers = headers)
    soup = BeautifulSoup(content.text,'html.parser')
    #find_all()方法返回的是一个列表
    new_url = soup.find_all('div',class_ = 'list-box m-t-n')
    news = []
    rule = '<a href="(.*?)" title=".*?">.*?</a>'
    new_all = re.findall(rule,str(list(new_url)[0]),re.I)
    for i in new_all:
        #获取新闻url，并补完整链接
        new_url = 'http://www.yaoyanbaike.com'+i
#         print(new_url)
        news.append(new_url)
#     print(news)
    return news #返回每个分类的全部全部文章链接，列表形式
def get_new_info(url,text_num,osdir_name):#获取一篇文章的详细内容
#     osdir_name = osdir_name[:-1]#去掉最后面的'_'字符
    content = requests.get(url,headers = headers)
    content.encoding='utf-8'#使用这个编码一般是可以的，防止出现乱码
    soup = BeautifulSoup(content.text,'html.parser')
    try:
         #记得使用Beautiful的find_all使用class查找的时候。由于class是内置类，不可以直接使用class,故class有一条下标的_记为class_
        #find()函数的返回一个元素,如果有多个，则返回第一个，使用find().get('属性名字')方法获取html元素中的标签如get('title')获取a元素中的title属性的值
        #find()函数一定要一层层的查找才可以找到，不可以跳过一层直接获取下两层的东西,要父-->子--->孙，不可父直接到孙
        new_title = soup.find('div',class_ = 'bg-white-only m-b wrapper').find('h1',class_='text-2x m-t-sm')#get_text()直接获取本标签的全部文本
        if new_title is None:
            pass
        else:
            new_title = new_title.get_text()
        #new_author = soup.find('div',class_ = 'bg-white-only m-b wrapper').find('div',class_='meta text-sm m-t').find('span',class_='m-r-sm text-success').get_text()
        #new_date = soup.find('div',class_ = 'bg-white-only m-b wrapper').find('div',class_='meta text-sm m-t').find('span',class_='text-muted time').get_text()
        new_content = soup.find('div',class_ = 'bg-white-only m-b wrapper').find('article',class_='content-text m-t-sm').get_text()
        article = new_content.replace(u'\n','')
    #     print(new_content.replace(u'\n',''))#去掉文章中所有换行符使用u''防止转义
    #     print(new_date[6:])
    #     print(new_author)
    #     print(new_title)
        #将文章写入txt文件中
        with open('e:py/谣言新闻/'+str(osdir_name)+'/'+str(text_num)+'.txt',mode='w', encoding='utf-8') as f:
            f.write(article)
            f.close()
            print('写好了一篇文章')
    except Exception as E:
        print('爬取错误！！！')
        raise #查看抛出的异常
#         print(E)
        print(url) 
#获取首页全部内容   
def get_index_info():
    start_url = 'http://www.yaoyanbaike.com/'
    content = requests.get(start_url,headers = headers,timeout = 500)#设置请求时间
    soup = BeautifulSoup(content.text,'html.parser')
 
    #记得使用Beautiful的find_all使用class查找的时候。由于class是内置类，不可以直接使用class,故class有一条下标的_记为class_
    new_title = soup.find_all('div',class_='media m-n wrapper-sm b-b')#返回的是一个列表
    
#     new_data = html.xpath('/html/body/section[1]/div/div[1]/div/div[2]/div/div[1]/div/p[1]')
#     print(new_title)
    for span in new_title:
#         print(span)
        new_title = span.find('a',class_='media-left hidden-xs').get('title')
        #find()函数的返回一个元素,如果有多个，则返回第一个，使用find().get('属性名字')方法获取html元素中的标签如get('title')获取a元素中的title属性的值
        #获取新闻url，并补完整链接
        new_url = start_url+span.find('a',class_='media-left hidden-xs').get('href')
#         print(new_url)
        #find()函数一定要一层层的查找才可以找到，不可以跳过一层直接获取下两层的东西,要父-->子--->孙，不可父直接到孙
        new_author = span.find('div',class_ = 'media-body').find('p',class_ = 'm-n l-h-2x').find('span',class_='m-r-xs text-success text-sm').get_text()
#         print(new_author)
        new_date = span.find('div',class_ = 'media-body').find('p',class_ = 'm-n l-h-2x').find('span',class_='time m-r-xs text-success text-sm').get_text()
#         print(new_date)    
    
'''
  正则匹配
  rule = '<a class="media-left hidden-xs" href="(.*?)" target="_blank" title="(.*?)">'#获取
    # ne = re.compile(rule)
    data = re.findall(rule,content.text,re.I)
    #提取首页每条新闻的详细链接
    data_url = []
    for i in range(0,len(data)):
        data_n = start_url+data[i][0]
        data_url.append(data_n)
#     print(data_url)
  
  ''' 

# get_index_url()
#封装成函数，开启多线程
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
        #获得每个分类下的20页
        all_list = all_news_info(item)
    #     print(all_list)
        osdir_name = item[:-1]
        os.makedirs('e:py/谣言新闻/'+str(osdir_name))
        #抓取每一个分类的每一页的全部链接
        c=0
        for i in all_list:
            b+=1
            news_list = get_news_url(i)#每一次返回每一页的20篇文章链接
            print(news_list)
            print('开始抓取分类的第'+str(b)+'页！')
            #遍历获取每一篇文章的详细内容
            for j in news_list:
                c+=1
                get_new_info(j,c,osdir_name)
            print('第'+str(c+20)+'篇文章抓取完成！')
            print('------------------------')
#     timer = threading.Timer(5*60*60,main)#设置线程
#     timer.start()#开启线程

#定义两个线程类，一人爬一半
class One_of_half(threading.Thread):#第一个线程，继承threading.Thread父类，要实现两个方法__inti__和run()
    def __init__(self,start_list_urls):
        threading.Thread.__init__(self)
        self.start_urls = start_list_urls
    
    #执行函数，开启线程
    def run(self):
        start_urls = self.start_urls
        main(start_urls)
    
#线程二
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
    
#     timer2 = threading.Timer(1,main)#再设置一个线程
#     timer2.start()#开启线程
    # news=get_news_url('http://www.yaoyanbaike.com/category/health.html')
# print(news)
# print(news)
# print(news)
# print(data)
# print(new_data)
# get_new_info('http://www.yaoyanbaike.com/a/836nv.html')