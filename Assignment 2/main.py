from tkinter import Tk, Canvas, Label, W, StringVar, Button
from tkinter.font import Font, BOLD
import loguru

from thread_test import myThread


class Square(Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        x_w = self.winfo_screenwidth()
        y_w = self.winfo_screenheight()
        ww = 900
        xx = 630
        x = (x_w - ww) / 2
        y = (y_w - xx) / 2
        self.geometry("%dx%d+%d+%d" % (ww, xx, x, y))
        self.configure(bg="#D1BA74")
        self.title('Rodeo')
        self.dict_rectangle = {}
        self.thread = None
        self.obstacles = [(5, 5), (5, 6), (5, 7), (6, 7), (7, 5), (7, 6), (7, 7)]
        self.X = (6, 6)
        self.walls = [0, 12]        # x must bu in [0, 12], y must be in [0, 12]
        self.bull_position = None
        self.robot_position = None
        self.ai_on_game = False

        self.title_label = Label(self, text="My First Rodeo", width=88, height=2, font=('MicroSoft Yahei', 12, BOLD), fg='white',
          bg="#F4606C")
        self.title_label.grid(row=0, column=0, columnspan=20, padx=5, pady=5)
        self.canvas = Canvas(self, width=520, height=530, bg="#D1BA74")

        for i in range(13):
            for j in range(13):
                if (i == 5 or i == 6 or i == 7) and (j == 5 or j == 6 or j == 7):
                    if i == 6 and j == 6:
                        # self.canvas.create_rectangle(i * 40, j * 40 + 5, (i + 1) * 40, j * 40 + 45, fill='#F4606C',
                        #                              outline='#D1BA74', width=5)
                        f = Font(size=30, family="Microsoft Yahei")
                        r = self.canvas.create_text((i * 40 + 20, j * 40 +24), text="X", font=f, tag=str(i)+str(j))
                        self.dict_rectangle[str(i) + str(j)] = r
                    elif i == 6 and j == 5:
                        r = self.canvas.create_rectangle(i * 40, j * 40 + 5, (i + 1) * 40, j * 40 + 45, fill='#F4606C',
                                                     outline='#D1BA74', width=5, tag=str(i)+str(j))
                        self.dict_rectangle[str(i) + str(j)] = r
                    else:
                        r = self.canvas.create_rectangle(i * 40, j * 40 + 5, (i + 1) * 40, j * 40 + 45, fill='black',
                                                     outline='#D1BA74', width=5, tag=str(i)+str(j))
                        self.dict_rectangle[str(i) + str(j)] = r
                else:
                    r = self.canvas.create_rectangle(i*40, j*40 + 5, (i+1)*40, j*40 + 45, fill='#F4606C', outline='#D1BA74',
                                                 width=5, tag=str(i)+str(j))
                    self.dict_rectangle[str(i)+str(j)] = r
        self.canvas.grid(row=1, column=1, columnspan=13, rowspan=13, padx=5, pady=12)

        self.label2 = Label(self, text="Menu", width=35, height=2, font=('MicroSoft Yahei', 12, BOLD), fg='white',
          bg="#F4606C")
        self.label2.grid(row=1, column=14, columnspan=2, rowspan=1, padx=5, pady=12)

        self.label3 = Label(self, text="Round Count", width=15, height=2, font=('MicroSoft Yahei', 12, BOLD), fg='white',
                            bg="#F4606C")
        self.label3.grid(row=2, column=14, columnspan=2, rowspan=1, padx=5, pady=12, sticky=W)

        self.round_count = StringVar()
        self.round_count.set("0")
        self.label4 = Label(self, textvariable=self.round_count, width=15, height=2, font=('MicroSoft Yahei', 12, BOLD),
                            fg='black',
                            bg="#19CAAD")
        self.label4.grid(row=2, column=15, columnspan=2, rowspan=1, padx=5, pady=12, sticky=W)

        self.bull_position_show = StringVar()
        self.bull_position_show.set("0, 0")
        self.robot_position_show = StringVar()
        self.robot_position_show.set("12, 12")

        self.label5 = Label(self, text="Bull Position", width=15, height=2, font=('MicroSoft Yahei', 12, BOLD),
                            fg='white',
                            bg="#F4606C")
        self.label5.grid(row=3, column=14, columnspan=2, rowspan=1, padx=5, pady=12, sticky=W)
        self.label5_1 = Label(self, textvariable=self.bull_position_show, width=15, height=2, font=('MicroSoft Yahei', 12, BOLD),
                            fg='black',
                            bg="#19CAAD")
        self.label5_1.grid(row=3, column=15, columnspan=2, rowspan=1, padx=5, pady=12, sticky=W)

        self.label6 = Label(self, text="Robot Position", width=15, height=2, font=('MicroSoft Yahei', 12, BOLD),
                            fg='white',
                            bg="#F4606C")
        self.label6.grid(row=4, column=14, columnspan=2, rowspan=1, padx=5, pady=12, sticky=W)
        self.label6_1 = Label(self, textvariable=self.robot_position_show, width=15, height=2,
                              font=('MicroSoft Yahei', 12, BOLD),
                              fg='black',
                              bg="#19CAAD")
        self.label6_1.grid(row=4, column=15, columnspan=2, rowspan=1, padx=5, pady=12, sticky=W)

        self.button_text = StringVar()
        self.button_text.set('Start Game')
        self.button1 = Button(self, textvariable=self.button_text, width=10, command=self.start_game,
           font=('MicroSoft Yahei', 12, BOLD),
           fg='Black', bg="#A0EEE1")
        self.button1.grid(row=6, column=14, columnspan=2, rowspan=1, padx=5, pady=12)

        self.button_text1 = StringVar()
        self.button_text1.set('Player')
        self.button2 = Button(self, textvariable=self.button_text1, width=10, command=self.ai_on,
                              font=('MicroSoft Yahei', 12, BOLD),
                              fg='Black', bg="#A0EEE1")
        self.button2.grid(row=5, column=14, columnspan=2, rowspan=1, padx=5, pady=12)

    def ai_on(self):
        if self.button_text1.get() == 'Player':
            self.button_text1.set("AI Robot")
            self.button2 = Button(self, textvariable=self.button_text1, width=10, command=self.ai_on,
                                  font=('MicroSoft Yahei', 12, BOLD),
                                  fg='Black', bg="#D6D5B7")
            self.button2.grid(row=5, column=14, columnspan=2, rowspan=1, padx=5, pady=12)
            self.ai_on_game = True
        else:
            self.button_text1.set("Player")
            self.button2 = Button(self, textvariable=self.button_text1, width=10, command=self.ai_on,
                                  font=('MicroSoft Yahei', 12, BOLD),
                                  fg='Black', bg="#A0EEE1")
            self.button2.grid(row=5, column=14, columnspan=2, rowspan=1, padx=5, pady=12)
            self.ai_on_game = False

    def start_game(self):
        if self.button_text.get() == "Start Game":
            self.thread = myThread(t)
            t.init_bull_position(0, 0)  # init bull's position (0, 0)
            t.init_robot_position(12, 12)  # initialize robot's position (12, 12)
            self.thread.start()
            self.button_text.set("Stop Game")
            self.button1 = Button(self, textvariable=self.button_text, width=10, command=self.start_game,
                                  font=('MicroSoft Yahei', 12, BOLD),
                                  fg='Black', bg="#D6D5B7")
            self.button1.grid(row=5, column=14, columnspan=2, rowspan=1, padx=5, pady=12)
            self.keyPressEvent()
        else:
            self.thread.killed = True
            self.button_text.set("Start Game")
            self.button1 = Button(self, textvariable=self.button_text, width=10, command=self.start_game,
                                  font=('MicroSoft Yahei', 12, BOLD),
                                  fg='Black', bg="#A0EEE1")
            self.button1.grid(row=5, column=14, columnspan=2, rowspan=1, padx=5, pady=12)

    def init_bull_position(self, pos_a, pos_b):
        self.canvas.delete(self.dict_rectangle[str(pos_a) + str(pos_b)])
        f = Font(size=30, family="Microsoft Yahei")
        r = self.canvas.create_text((pos_a * 40 + 20, pos_b * 40 + 24), text="B", font=f, tag=str(pos_a) + str(pos_b))
        self.bull_position = (pos_a, pos_b)
        self.dict_rectangle[str(pos_a) + str(pos_b)] = r
        self.bull_position_show.set(str(pos_a)+", " + str(pos_b))

    def init_robot_position(self, pos_a, pos_b):
        self.canvas.delete(self.dict_rectangle[str(pos_a) + str(pos_b)])
        f = Font(size=30, family="Microsoft Yahei")
        r = self.canvas.create_text((pos_a * 40 + 20, pos_b * 40 + 24), text="C", font=f, tag=str(pos_a) + str(pos_b))
        self.robot_position = (pos_a, pos_b)
        self.dict_rectangle[str(pos_a) + str(pos_b)] = r
        self.robot_position_show.set(str(pos_a) + ", " + str(pos_b))

    def bull_move(self, pos_a, pos_b):
        if (pos_a, pos_b) in self.walls:
            loguru.logger.warning("Bull collide with obstacles.")
        self.canvas.delete(self.dict_rectangle[str(self.bull_position[0]) + str(self.bull_position[1])])
        i = self.bull_position[0]
        j = self.bull_position[1]
        r = self.canvas.create_rectangle(i * 40, j * 40 + 5, (i + 1) * 40, j * 40 + 45, fill='#F4606C',
                                         outline='#D1BA74',
                                         width=5, tag=str(i) + str(j))
        self.dict_rectangle[str(i) + str(j)] = r
        self.init_bull_position(pos_a, pos_b)

    def robot_move(self, pos_a, pos_b):
        self.canvas.delete(self.dict_rectangle[str(self.robot_position[0]) + str(self.robot_position[1])])
        i = self.robot_position[0]
        j = self.robot_position[1]
        r = self.canvas.create_rectangle(i * 40, j * 40 + 5, (i + 1) * 40, j * 40 + 45, fill='#F4606C',
                                         outline='#D1BA74',
                                         width=5, tag=str(i) + str(j))
        self.dict_rectangle[str(i) + str(j)] = r
        self.init_robot_position(pos_a, pos_b)

    def change_round_count(self):
        self.round_count.set(str(int(self.round_count.get()) + 1))

    def set_round_count(self, count):
        self.round_count.set(str(count))

    def set_bull_position(self, pos_x, pos_y):
        self.bull_position_show.set(str(pos_x) + ", " + str(pos_y))

    def set_robot_position(self, pos_x, pos_y):
        self.robot_position_show.set(str(pos_x) + ", " + str(pos_y))

    def keyPressEvent(self) -> None:
        if not self.ai_on_game:
            self.bind('w', lambda event: self.player_move(self.robot_position[0], self.robot_position[1]-1))
            self.bind('s', lambda event: self.player_move(self.robot_position[0], self.robot_position[1]+1))
            self.bind('a', lambda event: self.player_move(self.robot_position[0]-1, self.robot_position[1]))
            self.bind('d', lambda event: self.player_move(self.robot_position[0]+1, self.robot_position[1]))
            self.bind('q', lambda event: self.player_move(self.robot_position[0] - 1, self.robot_position[1] - 1))
            self.bind('z', lambda event: self.player_move(self.robot_position[0] - 1, self.robot_position[1] + 1))
            self.bind('e', lambda event: self.player_move(self.robot_position[0] + 1, self.robot_position[1] - 1))
            self.bind('c', lambda event: self.player_move(self.robot_position[0] + 1, self.robot_position[1] + 1))

    def player_move(self, pos_x, pos_y):
        if self.position_is_valid(pos_x, pos_y) and self.collide_with_obstacles(pos_x, pos_y) and \
                self.on_X(pos_x, pos_y):
            if (pos_x, pos_y) == self.bull_position:
                loguru.logger.warning("Please try again, robot collied with bull.")
            else:
                self.robot_move(pos_x, pos_y)
        else:
            loguru.logger.warning("Please try again, " + str((pos_x, pos_y)) + " is invalid.")

    def position_is_valid(self, pos_x, pos_y):
        if self.walls[1] >= pos_x >= self.walls[0] and \
                self.walls[1] >= pos_y >= self.walls[0]:
            return True
        else:
            return False

    def collide_with_obstacles(self, pos_x, pos_y):
        if (pos_x, pos_y) in self.obstacles:
            loguru.logger.warning("collide with obstacles, " + str((pos_x, pos_y)))
            return False
        else:
            return True

    def on_X(self, pos_x, pos_y):
        if (pos_x, pos_y) == self.X:
            loguru.logger.warning("collide with X, " + str((pos_x, pos_y)))
            return False
        else:
            return True


if __name__ == '__main__':
    t = Square()
    t.mainloop()
