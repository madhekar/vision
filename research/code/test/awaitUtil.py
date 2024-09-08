import asyncio
import uuid
import inspect
from typing import Any, Awaitable, Callable, List, Union

def to_coroutine(f:Callable[..., Any]):
   async def wrapper(*args, **kwargs):
      return f(*args, **kwargs)
   return wrapper

def force_awaitable(function: Union[Callable[..., Awaitable[Any]], Callable[..., Any]]) -> Callable[..., Awaitable[Any]]:
   if inspect.iscoroutinefunction(function):
      return function
   else:
      return to_coroutine(function)
   

# You can return an async function with the original function object in its scope:

def to_coroutine(f:Callable[..., Any]):
   async def wrapper(*args, **kwargs):
      return f(*args, **kwargs)
   return wrapper

def force_awaitable(function: Union[Callable[..., Awaitable[Any]], Callable[..., Any]]) -> Callable[..., Awaitable[Any]]:
   if inspect.iscoroutinefunction(function):
      return function
   else:
      return to_coroutine(function)

# Now, if function is not awaitable, force_awaitable will return a coroutine function that contains function.

def test_sync(*args, **kwargs):
  return str(uuid.uuid4())

async def main():
   ret = await force_awaitable(test_sync)('url')
   print(ret)
asyncio.run(main())

