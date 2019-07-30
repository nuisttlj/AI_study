# 输入任意长度数字，倒序输出
# num = input("请输入任意长度的数字，系统会自动倒序输出：")
# num_dao = num[::-1]
# print(num_dao)

# 编写程序，提示输入整数X，然后计算从1到X连续整数的和
# num = int(input("请输入一个整数，系统会自动计算从1到该整数的值："))
# sum = 0
# for i in range(num+1):
#     print(i,end="")
#     sum += i
# print("结果为",sum)

# 有1、2、3、4个数字，能组成多少个互不相同且无重复数字的三位数，都是多少？
# sum = 0
# for i in range(1,5):
#     for j in range(1,5):
#         for k in range(1,5):
#             if i!=j and j!=k and k!=i:
#                 num = i*100+j*10+k
#                 print(num)
#                 sum += 1
# print("这样的数字有：",sum,"个")

# 打印所有的水仙花数
# for i in range(1,10):
# #     for j in range(1,10):
# #         for k in range(1,10):
# #             num = i*100+j*10+k
# #             if i**3+j**3+k**3 == num:
# #                 print(num)

# 输入某年某月某日，判断这一天是这一年的第几天

# 一个数如果恰好等于它的因子之和，这个数就成为“完数”，例如6=1+2+3，编程找出1000内的所有完数
# sum = 0
# for i in range(1,1001):
#     for j in range(1,i):
#         if i%j == 0:
#             sum += j
#             if sum == i:
#                 print(sum)

#handle方法
# class a:
#     def handle(self):
#         print("这是handle方法")
# class b(a):
#     def handle(self):
#         print("这是handle方法")
#         a.handle(self)
# a1 = a()
# b1 = b()
# b1.handle()

# 有一个计数器统计实例化了多少学生
# class student:
#     sum = 0
#     def __init__(self,name,sex):
#         self.name = name
#         self.sex = sex
#         student.sum += 1
#
# s1 = student("张三","男")
# s2 = student("张三","男")
# s3 = student("张三","男")
# s4 = student("张三","男")
# s5 = student("张三","男")
# s6 = student("张三","男")
# s7 = student("张三","男")
# print("共实例化了{}名学生".format(student.sum))

# 创建一个冰箱对象，实现储存食品，查询食品以及取出食品功能，要持久化储存

import os


class icebox:
    def __init__(self):
        self.input_str = input("第一次使用冰箱，请给它起个名字")
        self.welcome()
    def welcome(self):
        self.sig = input("欢迎使用{}冰箱！请输入对应序号执行操作：1.查询冰箱食物 2.放入食品 3.取出食品".format(self.input_str))
        if self.sig == "2":
            self.stoage()
        if self.sig == "1":
            self.search()
        if self.sig == "3":
            self.takeout()
    def stoage(self):
        while True:
            file = open("test1120.txt", 'a')
            self.food = input("放入什么食物")
            self.food_num = input("放入的食物数量")
            file.write(self.food)
            file.write(self.food_num)
            file.write("\n")
            file.close()
            print("操作成功")
            self.question = input("是否继续操作？（n结束，任意键继续）")
            if self.question == "n":
                print("感谢使用{}冰箱，欢迎下次再来！".format(self.input_str))
                break
            else:
                self.welcome()
    def search(self):
        while True:
            file = open("test1120.txt", 'r')
            str = file.read()
            print(str)
            self.question = input("是否继续操作？（n结束，任意键继续）")
            if self.question == "n":
                print("感谢使用{}冰箱，欢迎下次再来！".format(self.input_str))
                break
            else:
                self.welcome()
    def takeout(self):
        pass

i1 = icebox()
