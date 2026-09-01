# db_manager — PostgreSQL 数据库管理模块

对 PostgreSQL 常用操作进行封装的轻量模块。**仅使用当前项目环境已安装的库**：

| 依赖 | 版本参考 | 用途 |
|------|---------|------|
| [SQLAlchemy](https://www.sqlalchemy.org/) | 2.x | 连接池、反射、SQL 构建 |
| [psycopg](https://www.psycopg.org/) 3 | 3.x | PostgreSQL 驱动（`postgresql+psycopg://`） |
| `pgvector` / `langchain_postgres` | 可选 | 向量检索场景（本模块未强依赖） |

> 说明：本模块为**同步**封装，走 `postgresql+psycopg://`（psycopg3）。如需异步，可把
> `PGConfig.driver` 设为 `asyncpg`，但当前代码未提供异步接口。

---

## 目录结构

```
db_manager/
├── __init__.py        # 导出 PostgreSQLClient、PGConfig、build_dsn
├── pg_config.py       # 连接配置（PGConfig）+ DSN 生成
├── pg_client.py       # PostgreSQLClient：SQL 执行 / 建表 / CRUD / 事务
├── demo_usage.py      # 可直接运行的演示脚本
└── README.md
```

---

## 快速开始

### 1. 连接

```python
from db_manager import PostgreSQLClient, PGConfig, build_dsn

# 方式一：连接串
client = PostgreSQLClient(
    dsn="postgresql+psycopg://postgres:123456@localhost:5432/agent_embedding"
)

# 方式二：PGConfig
config = PGConfig(host="localhost", port=5432, database="postgres",
                  user="postgres", password="123456")
client = PostgreSQLClient(config=config)

# 方式三：从环境变量读取（PG_HOST / PG_PORT / PG_DATABASE / PG_USER / PG_PASSWORD ...）
client = PostgreSQLClient(config=PGConfig.from_env())
```

### 2. SQL 执行（任意语句）

```python
# DDL / DML：返回受影响行数
client.execute(
    "CREATE TABLE IF NOT EXISTS demo(id serial primary key, name text)"
)
client.execute("INSERT INTO demo(name) VALUES (:name)", {"name": "a"})
```

### 3. 查询

```python
all_rows  = client.fetch_all("SELECT * FROM demo WHERE id > :id", {"id": 0})   # List[dict]
one_row   = client.fetch_one("SELECT * FROM demo WHERE id = :id", {"id": 1})   # dict | None
some_rows = client.fetch_many("SELECT * FROM demo", size=10)
```

### 4. 建表 / 表结构

```python
# 原生 DDL 建表
client.create_table("""
    CREATE TABLE IF NOT EXISTS demo (
        id serial primary key,
        name text not null,
        created_at timestamptz default now()
    )
""")

client.table_exists("demo")                    # bool
client.list_tables()                           # List[str]
client.describe_table("demo")                  # 列信息
client.drop_table("demo")                      # 删除
client.drop_table("demo", cascade=True)        # 级联删除
```

### 5. 行级 CRUD

```python
# 插入（单行 / 多行）
client.insert("demo", {"name": "x"})
client.insert("demo", [{"name": "a"}, {"name": "b"}])
new_id = client.insert("demo", {"name": "c"}, returning="id")   # 返回自增主键

# 更新（where 传 dict 自动参数化）
client.update("demo", {"name": "y"}, {"id": 1})

# 删除
client.delete("demo", {"id": 1})

# 计数
client.count("demo")
client.count("demo", {"name": "y"})
```

`update` / `delete` / `count` 的 `where` 支持：`dict`（推荐）、原生 SQL 片段、
SQLAlchemy 表达式，参数化绑定，避免 SQL 注入。

### 6. 事务

```python
with client.transaction() as conn:
    conn.execute(text("INSERT INTO demo(name) VALUES (:n)"), {"n": "tx"})
    # 代码抛异常时整个事务自动回滚
```

---

## 环境变量参考

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `PG_HOST` | `localhost` | 数据库地址 |
| `PG_PORT` | `5432` | 端口 |
| `PG_DATABASE` / `PG_DB` | `postgres` | 数据库名 |
| `PG_USER` | `postgres` | 用户名 |
| `PG_PASSWORD` / `PG_PASS` | `""` | 密码 |
| `PG_SCHEMA` | `public` | 默认 schema |
| `PG_DRIVER` | `psycopg` | 驱动（`psycopg`/`psycopg2`/`asyncpg`） |
| `PG_POOL_SIZE` | `5` | 连接池大小 |
| `PG_MAX_OVERFLOW` | `10` | 溢出连接数 |
| `PG_ECHO` | `false` | 打印 SQL |

---

## 与项目现有代码的衔接

现有 `ai_app/main/deal_file/Rag_agent.py` 使用
`postgresql+psycopg://postgres:123456@localhost:5432/{db}` 的连接串配合
SQLAlchemy `create_engine` 和 `langchain_postgres`。本模块复用同一驱动与连接格式，
可作为常规 SQL/建表/CRUD 的统一入口，向量检索仍可继续使用 `langchain_postgres`。
