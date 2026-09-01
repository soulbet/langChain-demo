# -*- coding: utf-8 -*-
"""
db_manager
============

数据库管理模块：封装 PostgreSQL 常用操作。

依赖当前项目环境已安装的库：
    - SQLAlchemy 2.x
    - psycopg3（驱动，通过 ``postgresql+psycopg://`` 连接）
    - （可选）asyncpg / langchain_postgres 用于异步或向量场景

主要导出：
    - :class:`PostgreSQLClient`  PostgreSQL 客户端（连接池 / SQL 执行 / 建表 / CRUD / 事务）
    - :class:`PGConfig`          连接配置数据类，可从环境变量读取
    - :func:`build_dsn`          根据配置生成 SQLAlchemy 连接串
"""

from .pg_config import PGConfig, build_dsn, DEFAULT_PG_PORT
from .pg_client import PostgreSQLClient

__all__ = [
    "PostgreSQLClient",
    "PGConfig",
    "build_dsn",
    "DEFAULT_PG_PORT",
]
