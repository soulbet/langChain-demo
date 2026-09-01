# -*- coding: utf-8 -*-
"""
pg_client.py
==============

PostgreSQL 客户端封装（基于 SQLAlchemy 2.0 + psycopg3）。

功能概览：
    - 连接池管理（懒加载 engine，支持 pool_pre_ping / pool_size 等）
    - 任意 SQL 执行：``execute`` / ``fetch_all`` / ``fetch_one`` / ``fetch_many``
    - 建表 / 删表 / 判断表是否存在 / 列出表 / 描述表结构
    - 行级 CRUD：``insert`` / ``update`` / ``delete`` / ``count``
    - 事务上下文：``transaction``
    - 连接/引擎管理：``ping`` / ``close`` / ``dispose``

设计说明：
    - 依赖当前项目环境已有的库：SQLAlchemy 2.x、psycopg3（postgresql+psycopg://）。
    - ``_get_table`` 使用 ``autoload_with`` 反射表结构，INSERT/UPDATE/DELETE 走 SQL 语句构建，
      参数化绑定，避免 SQL 注入；且列名会经过数据库校验。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from sqlalchemy import MetaData, Table, and_, create_engine, delete, func, insert, inspect, select, text, update
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import NoSuchTableError, SQLAlchemyError

from .pg_config import PGConfig, build_dsn

logger = logging.getLogger(__name__)

# 允许作为 where 条件传入的类型
WhereClause = Union[Mapping[str, Any], str, Any]


class PostgreSQLClient:
    """PostgreSQL 客户端。

    参数（三选一，优先级 engine > dsn > config）：
        config :class:`PGConfig` 连接配置
        dsn   ``str``            直接给 SQLAlchemy 连接串（如 ``postgresql+psycopg://user:pass@host:port/db``）
        engine ``Engine``        直接复用已有 SQLAlchemy 引擎

    使用示例::

        client = PostgreSQLClient(dsn="postgresql+psycopg://postgres:123456@localhost:5432/mydb")
        client.execute("CREATE TABLE IF NOT EXISTS demo(id serial primary key, name text)")

        rows = client.fetch_all("SELECT * FROM demo WHERE id = :id", {"id": 1})
        client.insert("demo", {"name": "hello"})
        client.close()
    """

    def __init__(
        self,
        config: Optional[PGConfig] = None,
        dsn: Optional[str] = None,
        engine: Optional[Engine] = None,
        **kwargs: Any,
    ) -> None:
        # 允许 ``PostgreSQLClient(host=..., database=..., ...)`` 直接传连接参数
        if config is None and kwargs:
            config = PGConfig(**kwargs)

        if config is None:
            config = PGConfig()

        if engine is None and dsn is None:
            dsn = build_dsn(config)

        self.config = config
        self._dsn = dsn
        self._engine = engine

        # 元数据/表缓存
        self._metadata = MetaData()
        self._table_cache: Dict[tuple, Table] = {}

    # ------------------------------------------------------------------
    # 引擎管理
    # ------------------------------------------------------------------
    @property
    def engine(self) -> Engine:
        """懒加载的 SQLAlchemy 引擎。"""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    def _create_engine(self) -> Engine:
        engine_kwargs: Dict[str, Any] = {
            "pool_pre_ping": self.config.pool_pre_ping,
            "pool_size": self.config.pool_size,
            "max_overflow": self.config.max_overflow,
            "pool_timeout": self.config.pool_timeout,
            "echo": self.config.echo,
            "future": True,
        }
        engine_kwargs.update(self.config.engine_options or {})
        return create_engine(self._dsn, **engine_kwargs)

    def ping(self) -> bool:
        """测试数据库连接是否可用。成功返回 True，失败抛出异常。"""
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL 连接正常: %s", self.config.summary())
        return True

    def close(self) -> None:
        """关闭引擎，释放连接池。"""
        if self._engine is not None:
            self._engine.dispose()

    # ``dispose`` 是 ``close`` 的别名
    dispose = close

    # ------------------------------------------------------------------
    # 连接 / 事务上下文
    # ------------------------------------------------------------------
    @contextmanager
    def connect(self) -> Iterable[Connection]:
        """提供一个数据库连接（自动归还连接池）。

        注意：该上下文不会自动提交/回滚事务，适合只读或以 ``connection.commit()``
        显式提交的场景。写入操作推荐使用 :meth:`transaction`。
        """
        with self.engine.connect() as conn:
            yield conn

    @contextmanager
    def transaction(self, connection: Optional[Connection] = None) -> Iterable[Connection]:
        """提供事务上下文：正常结束自动 commit，异常自动 rollback。"""
        if connection is not None:
            with connection.begin():
                yield connection
        else:
            with self.engine.begin() as conn:
                yield conn

    # ------------------------------------------------------------------
    # 任意 SQL 执行
    # ------------------------------------------------------------------
    def execute(
        self,
        sql: str,
        params: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    ) -> int:
        """执行任意 SQL（DDL/DML 均可）。

        :param sql: SQL 语句，建议使用 ``:name`` 绑定参数，而非 f-string 拼接
        :param params: 绑定参数（dict 或 sequence）
        :return: 受影响行数（DDL 等语句返回 -1）

        示例::

            client.execute("INSERT INTO demo(name) VALUES (:name)", {"name": "a"})
            client.execute("UPDATE demo SET name=:name WHERE id=:id", {"id": 1, "name": "b"})
        """
        with self.engine.begin() as conn:
            result = conn.execute(text(sql), params or {})
            return result.rowcount

    def fetch_all(
        self,
        sql: str,
        params: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """执行查询，返回所有行（list[dict]）。"""
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return [dict(row) for row in result.mappings()]

    def fetch_one(
        self,
        sql: str,
        params: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """执行查询，返回第一行（dict），无结果返回 None。"""
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            row = result.mappings().first()
            return dict(row) if row is not None else None

    def fetch_many(
        self,
        sql: str,
        params: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
        size: int = 10,
    ) -> List[Dict[str, Any]]:
        """执行查询，返回最多 ``size`` 行。"""
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return [dict(row) for row in result.mappings().fetchmany(size)]

    # ``query`` 是 ``fetch_all`` 的别名
    query = fetch_all

    # ------------------------------------------------------------------
    # 建表 / 表结构
    # ------------------------------------------------------------------
    def create_table(self, ddl_sql: Optional[str] = None, table: Optional[Table] = None,
                     checkfirst: bool = True) -> bool:
        """创建表。

        :param ddl_sql: 原生 CREATE TABLE 语句
        :param table:   SQLAlchemy ``Table`` 对象（调用其 ``create``）
        :param checkfirst: 已存在是否跳过（默认 True）
        :return: 创建成功返回 True

        示例::

            # 方式一：原生 SQL
            client.create_table('''
                CREATE TABLE IF NOT EXISTS demo (
                    id serial primary key,
                    name text not null,
                    created_at timestamptz default now()
                )
            ''')

            # 方式二：SQLAlchemy Table 对象
            from sqlalchemy import Table, Column, Integer, String, MetaData
            t = Table("demo", MetaData(), Column("id", Integer, primary_key=True),
                      Column("name", String(100)))
            client.create_table(table=t)
        """
        if table is not None:
            table.create(self.engine, checkfirst=checkfirst)
            return True
        if ddl_sql:
            self.execute(ddl_sql)
            return True
        raise ValueError("create_table 需要提供 ddl_sql 或 table 参数")

    def table_exists(self, table_name: str, schema: Optional[str] = None) -> bool:
        """判断表是否存在。"""
        inspector = inspect(self.engine)
        return inspector.has_table(table_name, schema=schema or self.config.schema)

    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """列出指定 schema 下的所有表名。"""
        inspector = inspect(self.engine)
        return inspector.get_table_names(schema=schema or self.config.schema)

    def describe_table(self, table_name: str, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """描述表结构，返回列信息列表。"""
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name, schema=schema or self.config.schema)
        result: List[Dict[str, Any]] = []
        for col in columns:
            result.append(
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "default": col.get("default", None),
                    "primary_key": bool(col.get("primary_key", False)),
                    "autoincrement": col.get("autoincrement", False),
                }
            )
        return result

    def drop_table(self, table_name: str, schema: Optional[str] = None, cascade: bool = False) -> bool:
        """删除表。

        :param cascade: 是否级联（DROP TABLE ... CASCADE）
        """
        if not self.table_exists(table_name, schema):
            logger.warning("表不存在，跳过删除: %s.%s", schema or self.config.schema, table_name)
            return False
        ddl = f'DROP TABLE IF EXISTS "{schema or self.config.schema}"."{table_name}"'
        if cascade:
            ddl += " CASCADE"
        self.execute(ddl)
        return True

    # ------------------------------------------------------------------
    # 行级 CRUD
    # ------------------------------------------------------------------
    def _get_table(self, table_name: str, schema: Optional[str] = None) -> Table:
        """反射并缓存表对象。"""
        schema = schema or self.config.schema
        key = (schema, table_name)
        if key not in self._table_cache:
            try:
                self._table_cache[key] = Table(
                    table_name, self._metadata, schema=schema, autoload_with=self.engine
                )
            except NoSuchTableError as exc:
                raise NoSuchTableError(
                    f"表不存在: {schema}.{table_name}，请先 create_table"
                ) from exc
        return self._table_cache[key]

    def insert(
        self,
        table_name: str,
        data: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
        schema: Optional[str] = None,
        returning: Optional[Union[str, Sequence[str]]] = None,
    ) -> Any:
        """插入单行或多行。

        :param data: dict（单行）或 list[dict]（多行）
        :param returning: 指定要返回的列名（如 ``["id"]``），返回对应行
        :return: 未指定 returning 时返回受影响行数；否则返回值行
        """
        table = self._get_table(table_name, schema)
        stmt = insert(table)
        if returning:
            if isinstance(returning, str):
                returning = [returning]
            stmt = stmt.returning(*[table.c[col] for col in returning])
        with self.engine.begin() as conn:
            result = conn.execute(stmt, data)
            if returning:
                rows = [dict(row) for row in result.mappings().all()]
                return rows[0] if isinstance(data, Mapping) and rows else rows
            # 部分驱动对多值 INSERT 返回 rowcount=-1，回退到按 data 长度估算
            rc = result.rowcount
            if rc == -1 and isinstance(data, Sequence):
                rc = len(data)
            return rc

    def update(
        self,
        table_name: str,
        set_values: Mapping[str, Any],
        where: WhereClause,
        schema: Optional[str] = None,
    ) -> int:
        """更新符合条件的数据。

        :param set_values: 要更新的列->值
        :param where: 条件。dict（如 ``{"id": 1}``）或 SQLAlchemy 表达式，或原生 SQL 片段
        :return: 受影响行数
        """
        table = self._get_table(table_name, schema)
        stmt = update(table).values(**set_values).where(self._build_where(table, where))
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            return result.rowcount

    def delete(self, table_name: str, where: WhereClause, schema: Optional[str] = None) -> int:
        """删除符合条件的数据。

        :param where: 条件，同 :meth:`update`
        :return: 受影响行数
        """
        table = self._get_table(table_name, schema)
        stmt = delete(table).where(self._build_where(table, where))
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            return result.rowcount

    def count(self, table_name: str, where: Optional[WhereClause] = None,
              schema: Optional[str] = None) -> int:
        """统计行数。"""
        table = self._get_table(table_name, schema)
        stmt = select(func.count()).select_from(table)
        if where is not None:
            stmt = stmt.where(self._build_where(table, where))
        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            return result.scalar_one()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _build_where(table: Table, where: WhereClause) -> Any:
        """把 where 参数归一化为 SQLAlchemy 条件表达式。"""
        if isinstance(where, Mapping):
            conds = [table.c[k] == v for k, v in where.items()]
            return and_(*conds) if conds else and_(True)
        if isinstance(where, str):
            return text(where)
        # 传入的应是 SQLAlchemy 表达式
        return where
