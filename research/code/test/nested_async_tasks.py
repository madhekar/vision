import asyncio
import os

class EmbeddingClass:
    async def master_method(self, uri):
      tasks = [self.sub_method(f) for f in os.listdir(uri)]
      results = await asyncio.gather(*tasks)
      print(results) 

    async def sub_method(self, uri):
      subtasks = [self.generateId(), self.timestamp(uri), self.locationDetails(uri), self.namesOfPeople(uri), self.describeImage(uri) ]
      results = await asyncio.gather(*subtasks)
      print(results) 

    async def generateId(self):
      return "generateId done"  
    
    async def timestamp(self, uri):
      return "timestamp done: "   
    
    async def locationDetails(self, uri):
      return "locationDetails done: "   
    
    async def namesOfPeople(self, uri):
      return "namesOfPeople done: "

    async def describeImage(self,uri):
      return "describeImage done: "
    
em = EmbeddingClass()

asyncio.run(em.master_method('.'))
print('do normal stuff..')