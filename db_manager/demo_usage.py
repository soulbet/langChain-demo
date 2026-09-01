# -*- coding: utf-8 -*-
"""
demo_usage.py
===============

PostgreSQLClient 使用示例（可运行）。

本示例默认使用项目约定地址 ``localhost:5432``、数据库 ``postgres``、用户 ``postgres``，
密码为空（可通过环境变量或 ``PGConfig`` 覆盖）。若无本地 PostgreSQL 服务，可把
``DEMO_DSN`` 改为你自己的数据库连接串。

运行方式::

    python db_manager/demo_usage.py
"""

import sys
from pathlib import Path

# 让脚本可直接在项目根目录下运行（把项目根加入 sys.path）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from db_manager import PostgreSQLClient, PGConfig, build_dsn  # noqa: E402


def main() -> None:
    # ---------------- 1. 连接 ----------------
    # 方式一：使用 PGConfig 构造
    config = PGConfig(
        host="localhost",
        port=5432,
        database="postgres",
        user="postgres",
        password="123456",   # 按实际环境修改
    )
    dsn = build_dsn(config)

    print("DSN:", dsn)

    # 方式二：直接传连接串（推荐，直观）
    client = PostgreSQLClient(dsn=dsn)

    # 方式三：从环境变量读取（PG_HOST/PG_PORT/...）
    # client = PostgreSQLClient(config=PGConfig.from_env())

    # ---------------- 2. 建表 ----------------
    client.create_table("""
        CREATE TABLE IF NOT EXISTS demo_user (
            id serial primary key,
            name text not null,
            age int,
            email text,
            created_at timestamptz default now()
        )
    """)
    print("表存在:", client.table_exists("demo_user"))

    # ---------------- 3. 插入数据 ----------------
    client.insert(
        "demo_user",
        [
            {"name": "Alice", "age": 20, "email": "alice@example.com"},
            {"name": "Bob", "age": 25, "email": "bob@example.com"},
        ],
    )
    # 返回自增主键
    new_id = client.insert(
        "demo_user", {"name": "Carol", "age": 30, "email": "carol@example.com"},
        returning="id",
    )
    print("插入了新 id:", new_id)

    # ---------------- 4. 查询 ----------------
    rows = client.fetch_all(
        "SELECT id, name, age FROM demo_user WHERE age >= :age ORDER BY id",
        {"age": 20},
    )
    print("查询结果:", rows)

    one = client.fetch_one("SELECT name FROM demo_user WHERE id = :id", {"id": 1})
    print("首行:", one)

    # ---------------- 5. 更新 / 删除 / 计数 ----------------
    affected = client.update("demo_user", {"age": 21}, {"name": "Alice"})
    print("更新行数:", affected)

    total = client.count("demo_user")
    print("总行数:", total)

    deleted = client.delete("demo_user", {"name": "Bob"})
    print("删除行数:", deleted)

    # ---------------- 6. 描述表结构 ----------------
    cols = client.describe_table("demo_user")
    print("表结构:", cols)

    # ---------------- 7. 事务 ----------------
    with client.transaction() as conn:
        conn.execute(
            text("INSERT INTO demo_user(name, age) VALUES (:name, :age)"),
            {"name": "Dave", "age": 40},
        )
        # 若此处抛异常，上面的插入会自动回滚

    # ---------------- 8. 清理 ----------------
    client.drop_table("demo_user")
    print("表存在(清理后):", client.table_exists("demo_user"))

    client.close()
    print("演示完成 ✅")


if __name__ == "__main__":
    main()
