from typing import Optional

from fastapi import Depends, FastAPI

from sqlmodel import Field, Session, SQLModel, create_engine

# -------------------
# 1. 模型（自动 = 表结构 + 接口文档）
# -------------------
class Hero(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    secret_name: str

# -------------------
# 2. 数据库连接（最佳实践依赖注入）
# -------------------
engine = create_engine("sqlite:///database.db")

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session  # 提供 + 自动关闭
app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()
"""
yield 在这里不是生成器迭代，是 FastAPI 利用生成器特性实现的 “两段式执行”。
yield原理：
依赖函数遇到 yield 会暂停执行，把 yield 后面的值返回出去。
FastAPI 拿到这个值，传给接口去处理业务。
等请求完全结束（不管成功还是报错），FastAPI 再唤醒这个依赖函数。
函数从暂停位置继续往下跑，执行 yield 后面的清理代码。
"""
@app.get("/heroes/")
def get_heroes(session: Session = Depends(get_session)):
    return session.query(Hero).all()

@app.post("/heroes/")
def create_hero(hero: Hero, session: Session = Depends(get_session)):
    session.add(hero)
    session.commit()
    return hero