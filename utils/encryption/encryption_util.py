#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/8/27 14:44
@Author : zyf
@File : encryption_util.py
@Project : langChain-demo
@Software : PyCharm
@explain :
@DESCRIPTION :
"""
from beartype._data import cls

from db_manager import PGConfig, build_dsn, PostgreSQLClient
from utils.encryption.encryption import generate_key, encrypt_str, decrypt_str


class EncryptionUtil(object):

    def __init__(self):
        pass

    def pg_client(self,) -> PostgreSQLClient:
        config = PGConfig(
                host="localhost",
                port=5432,
                database="info_manager",
                schema="config",
                user="postgres",
                password="123456",  # 按实际环境修改
        )

        dsn = build_dsn(config)
        print("DSN:", dsn)

        # 方式二：直接传连接串（推荐，直观）
        client = PostgreSQLClient(dsn=dsn,config=config)
        return client

    def save_info(self, name: str,save_text:str,url:str) -> str:
        """

        :param data:
        """
        client=self.pg_client()
        name_row=client.fetch_one(f"""
             select * from config.passwd where name = '{name}'
        """)
        key = generate_key()
        if not name_row:

            client.insert(
                    "passwd",
                    [
                            {"name": name, "key": key, "encrytion": encrypt_str(save_text, key),
                             "describe": name,"url":url},
                    ],
            )
        else:
            client.update(
                    "passwd",{"name": name, "key": key,
                              "encrytion": encrypt_str(save_text, key),
                             "describe": name,"url":url},{"name": name}
            )
            print(f"{name}:已更新")
    def get_decrypt_str(self,name: str) -> str:
        client = self.pg_client()
        name_row = client.fetch_one(f"""
                     select * from config.passwd where name = '{name}'
                """)
        key=name_row["key"]
        encrytion=name_row["encrytion"]
        org_text=decrypt_str(encrytion,bytes.fromhex(key.replace(r"\x", "")))
        return org_text



if __name__ == "__main__":
    EncryptionUtil().get_decrypt_str("ds_api_key")