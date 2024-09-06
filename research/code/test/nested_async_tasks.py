import asyncio
import glob
import os
import uuid
from random import randint

sem = asyncio.Semaphore(2)

def getRecursive(rootDir):
    f_list=[]
    for fn in glob.glob(rootDir + '/**/*', recursive=True):
        if not os.path.isdir(os.path.abspath(fn)):
            f_list.append(os.path.abspath(fn))
    return f_list 

class EmbeddingClass:

    async def safeMaster(self, iList):
      async with sem:  # semaphore limits num of simultaneous downloads
        return await self.master_method(iList)

    async def master_method(self, imgList):
      #async with sem:
         tasks = [asyncio.ensure_future(self.sub_method(f)) for f in imgList]
         return await asyncio.gather(*tasks)

    async def sub_method(self, uri):
         subtasks = [asyncio.ensure_future(self.generateId()), 
                  asyncio.ensure_future(self.timestamp(uri)), 
                  asyncio.ensure_future(self.locationDetails(uri)), 
                  asyncio.ensure_future(self.namesOfPeople(uri)), 
                  asyncio.ensure_future(self.describeImage(uri)) ]
         return await asyncio.gather(*subtasks)


    async def generateId(self):
      return { 'id': str(uuid.uuid4()) }  
    
    async def timestamp(self, uri):
      wait_time = randint(1, 3)
      await asyncio.sleep(wait_time)
      return f"timestamp done."   
    
    async def locationDetails(self, uri):
      wait_time = randint(1, 3)
      await asyncio.sleep(wait_time)
      return f"locationDetails done."   
    
    async def namesOfPeople(self, uri):
      wait_time = randint(3, 7)
      await asyncio.sleep(wait_time)
      return f"namesOfPeople done."

    async def describeImage(self,uri):
      wait_time = randint(5, 10)
      await asyncio.sleep(wait_time)
      return f"describeImage done."
    
em = EmbeddingClass()
iList = getRecursive('/Users/emadhekar/erase_me/images/')
r = asyncio.run(em.safeMaster(iList))
print('do normal stuff..', r)