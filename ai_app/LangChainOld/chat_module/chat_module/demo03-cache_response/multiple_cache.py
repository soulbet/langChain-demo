from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from ai_app.model_factory.model_factory import model_factory

llm=model_factory().create_model()

from langchain_core.caches import InMemoryCache

set_llm_cache(InMemoryCache())

# The first time, it is not yet in cache, so it should take longer
llm.invoke("Tell me a joke")

# SQLite 缓存，也可使用其他数据库缓存
set_llm_cache(SQLiteCache(database_path=".langchain.db"))