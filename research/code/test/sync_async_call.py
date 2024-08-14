import asyncio
import threading

_loop = asyncio.new_event_loop()

_thr = threading.Thread(target=_loop.run_forever, name= 'zesha_async_runnder', daemon=True)

def run_async(coroutine):
    if not _thr.is_alive():
        _thr.start()
        future = asyncio.run_coroutine_threadsafe(coroutine, _loop)
        return future.result()
    
if __name__=="__main__":
    async def hel():
        await asyncio.sleep(20)
        print('running in thread', threading.current_thread())    
        return 400
    
    def i():
        y = run_async(hel())
        print('answer: ', y, threading.current_thread())

    async def h():
        i()

    asyncio.run(h())        

    print('zesha')