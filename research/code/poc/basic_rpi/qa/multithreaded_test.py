import requests
import time
from queue import Queue
from threading import Thread

# prepare urls 
que = Queue()
for i in range(500):
  que.put(('https://picsum.photos/id/{}/100/100'.format(i +1), i+1))
  
def get_files():
  sess = requests.session()
  while( not que.empty()):
    url = que.get()
    print('url:',url)
    r=sess.get(url[0])
    with open('./images/{}.png'.format(url[1]), 'wb') as f:
       f.write(r.content)
    print('Downloaded Picture {} using multiple threads.'.format(url[1]))

threads=[]
for i in range(15):
  threads.append(Thread(target=get_files))

start=time.time()
for i in threads:
    i.start()
for i in threads:
    i.join()
end=time.time()-start

print('time taken: {}'.format(end))
