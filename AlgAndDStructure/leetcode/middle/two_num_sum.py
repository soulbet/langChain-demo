#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/9/1 10:35
@Author : zyf
@File : two_num_add.py
@Project : langChain-demo
@Software : PyCharm
@explain :
@DESCRIPTION :
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
请你将两个数相加，并以相同形式返回一个表示和的链表。
你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    # ---------- 方法一：递归解法（你提供的优雅写法） ----------
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode], carry=0) -> Optional[ListNode]:
        # 递归边界：两个链表都为空且没有进位时结束
        if l1 is None and l2 is None and carry == 0:
            return None

        # 计算当前位的和
        s = carry
        if l1:
            s += l1.val
            l1 = l1.next
        if l2:
            s += l2.val
            l2 = l2.next

        # 创建当前节点，并递归计算下一位
        return ListNode(s % 10, self.addTwoNumbers(l1, l2, s // 10))

    # ---------- 方法二：迭代解法（更直观，避免递归深度限制） ----------
    def addTwoNumbersIterative(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)  # 虚拟头节点
        current = dummy
        carry = 0

        while l1 or l2 or carry:
            s = carry
            if l1:
                s += l1.val
                l1 = l1.next
            if l2:
                s += l2.val
                l2 = l2.next

            current.next = ListNode(s % 10)
            current = current.next
            carry = s // 10

        return dummy.next


# ---------- 辅助函数：将列表转换为链表 ----------
def create_linked_list(arr):
    dummy = ListNode(0)
    current = dummy
    for val in arr:
        current.next = ListNode(val)
        current = current.next
    return dummy.next


# ---------- 辅助函数：将链表转换为列表（便于打印） ----------
def linked_list_to_list(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result


# ---------- 测试代码 ----------
if __name__ == "__main__":
    # 输入: l1 = [2,4,3] (342), l2 = [5,6,4] (465)
    l1 = create_linked_list([2, 4, 3])
    l2 = create_linked_list([5, 6, 4])

    sol = Solution()

    # 测试递归解法
    result_rec = sol.addTwoNumbers(l1, l2)
    print("递归解法结果:", linked_list_to_list(result_rec))  # 输出: [7, 0, 8]

    # 测试迭代解法（需要重新创建链表，因为上面的递归已经修改了 l1, l2 的指针）
    l1 = create_linked_list([2, 4, 3])
    l2 = create_linked_list([5, 6, 4])
    result_iter = sol.addTwoNumbersIterative(l1, l2)
    print("迭代解法结果:", linked_list_to_list(result_iter))  # 输出: [7, 0, 8]