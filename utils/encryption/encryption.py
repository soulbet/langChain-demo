#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/8/27 10:35
@Author : zyf
@File : encryption.py
@Project : langChain-demo
@Software : PyCharm
@explain :
@DESCRIPTION :
"""

from cryptography.fernet import Fernet


def generate_key() -> bytes:
    """生成密钥（注意保存好，密钥丢失则无法解密）"""
    return Fernet.generate_key()


def encrypt_str(text: str, key: bytes) -> str:
    """加密字符串 -> 密文"""
    f = Fernet(key)
    return f.encrypt(text.encode("utf-8")).decode("utf-8")


def decrypt_str(cipher_text: str, key: bytes) -> str:
    """解密密文 -> 原始字符串"""
    f = Fernet(key)
    return f.decrypt(cipher_text.encode("utf-8")).decode("utf-8")


if __name__ == "__main__":
    key = generate_key()
    print(f"密钥: {key.decode()}")

    original = "hello world"
    encrypted = encrypt_str(original, key)
    decrypted = decrypt_str(encrypted, key)

    print(f"原文: {original}")
    print(f"密文: {encrypted}")
    print(f"解密: {decrypted}")

