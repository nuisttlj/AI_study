import os


class Fridge:
    def __init__(self):
        self.fridge_name = input("第一次使用冰箱，请给它起个名字：")
        self.path = r"D:\fridge"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.fridge_path = os.path.join(self.path, "{0}.txt".format(self.fridge_name).format(self.fridge_name))
        fridge_file = open(self.fridge_path, "a")
        fridge_file.close()
        while True:
            choose = int(input("欢迎使用{0}冰箱，请输入对应序号执行操作：1.查询冰箱食物 2.放入食品 3.取出食品：".format(self.fridge_name)))
            if choose == 1:
                print(self.query_food())
            elif choose == 2:
                print(self.input_food())
            elif choose == 3:
                print(self.output_food())
            else:
                print("您的输入有误，请重新输入")
            choose1 = input("是否继续操作? （n结束，任意键继续）")
            if choose1 == "n":
                print("感谢您使用{0}冰箱，欢迎下次再来!".format(self.fridge_name))
                break
            else:
                continue

    def query_food(self):
        food_list = open(self.fridge_path, "r").read().splitlines()
        return food_list

    def input_food(self):
        new_food = input("请输入要放入的食品名称：")
        num = int(input("请输入要放入的食品数量："))
        food_file = open(self.fridge_path, "a")
        for _ in range(num):
            food_file.write("{0}\n".format(new_food))
        food_file.close()
        return open(self.fridge_path, "r").read().splitlines()

    def output_food(self):
        output_food = input("请输入要取出的食品名称：")
        num = int(input("请输入要取出的食品数量："))
        fridge_file = open(self.fridge_path, "r")
        food_list = fridge_file.read().splitlines()
        count = food_list.count(output_food)
        if count < num:
            print("冰箱里没有足够的{0}".format(output_food))
        else:
            for _ in range(num):
                food_list.remove(output_food)
        fridge_file = open(self.fridge_path, "w")
        for i in food_list:
            fridge_file.write("{0}\n".format(i))
        fridge_file.close()
        return food_list


if __name__ == '__main__':
    fridge = Fridge()
