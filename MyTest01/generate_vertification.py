import PIL.Image as image
import PIL.ImageDraw as draw
import PIL.ImageFont as font
import random
import os


class vertication_generator:
    def __init__(self):
        self.width = 120
        self.height = 40
        self.my_font = font.truetype(r"msyh.ttf", size=30)

    def generate_panel(self):
        self.panel = image.new('RGB', (self.width, self.height), (255, 255, 255))
        # self.panel.show()
        return self.panel

    def generate_ran_letter(self):
        self.num = random.randint(0, 9)
        self.letter = chr(random.randint(97, 122))
        self.Letter = chr(random.randint(65, 90))
        self.strings = str(random.choice([self.num, self.letter, self.Letter]))
        # print(self.strings)
        return self.strings
        # return str(self.num)

    def generate_ran_bgcolor(self):
        self.bgcolor1 = random.randint(64, 255)
        self.bgcolor2 = random.randint(64, 255)
        self.bgcolor3 = random.randint(64, 255)
        return (self.bgcolor1, self.bgcolor2, self.bgcolor3)

    def generate_ran_lettercolor(self):
        self.lettercolor1 = random.randint(32, 128)
        self.lettercolor2 = random.randint(32, 128)
        self.lettercolor3 = random.randint(32, 128)
        return (self.lettercolor1, self.lettercolor2, self.lettercolor3)

    def generate_ran_axis(self):
        self.startx = random.randint(10, 229)
        self.starty = random.randint(10, 69)
        self.endx = random.randint(self.startx + 1, 230)
        self.endy = random.randint(self.starty + 1, 70)
        return (self.startx, self.starty, self.endx, self.endy)


if __name__ == '__main__':
    vt_generator = vertication_generator()
    # print(vt_generator.generate_ran_axis())
    # vt_generator.generate_panel()
    # vt_generator.generate_ran_letter()
    for num in range(200):
        panel = vt_generator.generate_panel()
        my_draw = draw.Draw(panel)
        for i in range(vt_generator.width):
            for j in range(vt_generator.height):
                my_draw.point((i, j), fill=vt_generator.generate_ran_bgcolor())
        # panel.show()
        letter = ""
        for k in range(6):
            gen_letter = vt_generator.generate_ran_letter()
            letter += gen_letter
            my_draw.text((k * 20 + 5, 5), text=gen_letter,
                         fill=vt_generator.generate_ran_lettercolor(), font=vt_generator.my_font)
        my_draw.rectangle(vt_generator.generate_ran_axis(), outline="red")
        my_draw.line(vt_generator.generate_ran_axis(), fill="blue", width=3)
        my_draw.line(vt_generator.generate_ran_axis(), fill="yellow", width=4)
        my_draw.line(vt_generator.generate_ran_axis(), fill="black", width=1)
        my_draw.arc(vt_generator.generate_ran_axis(), 0, 360, fill="green")
        if not os.path.exists(r"C:\vertification_test"):
            os.makedirs(r"C:\vertification_test")
        panel.save(r"C:\vertification_test\{0}.jpg".format(letter))
