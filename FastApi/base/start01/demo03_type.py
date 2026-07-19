from typing import Union


## 简单类型声明
def get_items(item_a: str, item_b: int, item_c: float, item_d: bool, item_e: bytes):
    return item_a, item_b, item_c, item_d, item_e

## 列表声明
def process_items(items: list[str]):
    for item in items:
        print(item)

## 元组和集合
def process_items_1(items_t: tuple[int, int, str], items_s: set[bytes]):
    return items_t, items_s

## 字典
def process_items_2(prices: dict[str, float]):
    for item_name, item_price in prices.items():
        print(item_name)
        print(item_price)

## 联合类型,声明一个参数可以是多个类型
def process_items_3(items_t: Union[str,int]):
    return items_t

## 类作为类型
class Person:
    def __init__(self, name: str):
        self.name = name

def get_person_name(person: Person):
    return person.name


## Pydantic 数据验证

