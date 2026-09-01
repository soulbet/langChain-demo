# -*- coding: utf-8 -*-
"""
pg_config.py
==============

PostgreSQL 连接配置。

提供 ``PGConfig`` 数据类，支持：
    - 直接传参构造 ``PGConfig(host=..., database=..., ...)``
    - 从环境变量读取（前缀 ``PG_``，如 ``PG_HOST/PG_PORT/PG_DATABASE/PG_USER/PG_PASSWORD/PG_SCHEMA``）
    - 生成 SQLAlchemy 连接串 ``build_dsn(config)`` / ``config.to_dsn()``

默认值与项目现有约定保持一致（host=localhost、port=5432、driver=psycopg）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus

DEFAULT_PG_PORT = 5432

# SQLAlchemy 支持的 PostgreSQL 驱动
VALID_DRIVERS = {"psycopg", "psycopg2", "asyncpg"}


@dataclass
class PGConfig:
    """PostgreSQL 连接配置。

    属性说明：
        host         数据库地址，默认 ``localhost``
        port         数据库端口，默认 ``5432``
        database     数据库名，默认 ``postgres``
        user         用户名，默认 ``postgres``
        password     密码，默认空
        schema       默认 schema，默认 ``public``
        driver       SQLAlchemy 驱动，默认 ``psycopg``（psycopg3）
        pool_size    连接池大小，默认 5
        max_overflow 连接池溢出大小，默认 10
        pool_timeout 获取连接超时（秒），默认 30
        pool_pre_ping 连接前 ping，默认 True
        echo         是否打印 SQL 日志，默认 False
    """

    host: str = "localhost"
    port: int = DEFAULT_PG_PORT
    database: str = "postgres"
    user: str = "postgres"
    password: str = ""
    schema: str = "public"
    driver: str = "psycopg"

    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_pre_ping: bool = True
    echo: bool = False
    # 额外的 SQLAlchemy 连接参数（例如 connect_args、application_name 等）
    engine_options: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.driver not in VALID_DRIVERS:
            raise ValueError(
                f"不支持的驱动 driver={self.driver!r}，可选其一：{sorted(VALID_DRIVERS)}"
            )

    # ------------------------------------------------------------------
    # 从环境变量读取
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, prefix: str = "PG_") -> "PGConfig":
        """从环境变量构造配置。

        环境变量名（带前缀 ``prefix``，默认 ``PG_``）：
            PG_HOST / PG_PORT / PG_DATABASE / PG_USER / PG_PASSWORD /
            PG_SCHEMA / PG_DRIVER / PG_POOL_SIZE / PG_MAX_OVERFLOW ...

        未设置的环境变量沿用对应默认值。
        """
        return cls(
            host=os.getenv(f"{prefix}HOST", "localhost"),
            port=int(os.getenv(f"{prefix}PORT", DEFAULT_PG_PORT)),
            database=os.getenv(f"{prefix}DATABASE", os.getenv(f"{prefix}DB", "postgres")),
            user=os.getenv(f"{prefix}USER", "postgres"),
            password=os.getenv(f"{prefix}PASSWORD", os.getenv(f"{prefix}PASS", "")),
            schema=os.getenv(f"{prefix}SCHEMA", "public"),
            driver=os.getenv(f"{prefix}DRIVER", "psycopg"),
            pool_size=int(os.getenv(f"{prefix}POOL_SIZE", "5")),
            max_overflow=int(os.getenv(f"{prefix}MAX_OVERFLOW", "10")),
            pool_timeout=int(os.getenv(f"{prefix}POOL_TIMEOUT", "30")),
            pool_pre_ping=os.getenv(f"{prefix}POOL_PRE_PING", "true").lower() == "true",
            echo=os.getenv(f"{prefix}ECHO", "false").lower() == "true",
        )

    # ------------------------------------------------------------------
    # 生成连接串
    # ------------------------------------------------------------------
    def to_dsn(self) -> str:
        """生成 SQLAlchemy 连接串。

        形如：``postgresql+psycopg://user:pass@host:port/database``

        用户名/密码会自动做 URL 编码，避免特殊字符报错。
        """
        return build_dsn(self)

    def summary(self) -> str:
        """返回不含密码的简要信息，便于日志打印。"""
        return (
            f"{self.driver}://{self.user}@{self.host}:{self.port}/{self.database}"
            f" (schema={self.schema})"
        )


def build_dsn(config: PGConfig) -> str:
    """根据配置生成 SQLAlchemy 连接串。"""
    print("当前打印的 config 的 id:", id(config))
    user = quote_plus(config.user)
    password = quote_plus(config.password)
    return (
        f"postgresql+{config.driver}://{user}:{password}"
        f"@{config.host}:{config.port}/{config.database}"
    )
