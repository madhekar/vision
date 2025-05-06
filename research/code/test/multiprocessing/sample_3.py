from datetime import datetime
import concurrent.futures

def process(num_jobs=1,**kwargs) :

    from functools import partial
   
    iterobj = range(num_jobs)
    args = []
    func = globals()['test_multi']

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as ex:
        ## using map
        result = ex.map(partial(func,*args,**kwargs),iterobj)

    return result

def test_multi(*args,**kwargs):

    starttime = datetime.utcnow()
    iternum = args[-1]
    test = []
    for i in range(200000):
        test = test + [i]
    
    return iternum, (datetime.utcnow()-starttime)


if __name__ == '__main__' :

    max_processors = 10
    
    for i in range(max_processors):      
        starttime = datetime.utcnow()
        result = process(i+1)
        finishtime = datetime.utcnow()-starttime
        if i == 0:
            chng = 0
            total = 0
            firsttime = finishtime
        else:
            chng = finishtime/lasttime*100 - 100
            total = finishtime/firsttime*100 - 100
        lasttime = finishtime
        print(f'Multi took {finishtime} for {i+1} processes changed by {round(chng,2)}%, total change {round(total,2)}%')

