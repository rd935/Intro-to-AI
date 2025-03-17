import random
import threading, time, loguru

import tkinter as tk
from tkinter import messagebox


class myThread(threading.Thread):  # threading.Thread
    def __init__(self, root):
        threading.Thread.__init__(self)
        self.root = root
        self.killed = False
        self.random = random.Random()
        self.target_position = (6, 4)
        self.target_yes = False
        self.target_yes1 = False
        self.continue_game = True

    def run(self):
        while self.continue_game:
            time.sleep(1)
            if self.target_yes:
                self.root.robot_move(self.root.robot_position[0]+1, self.root.robot_position[1]+1)
                self.target_yes = False
            elif self.target_yes1:
                self.root.robot_move(self.root.robot_position[0] - 1, self.root.robot_position[1] + 1)
                self.target_yes1 = False
            elif self.root.bull_position == (6, 5):
                self.root.robot_move(self.root.robot_position[0], self.root.robot_position[1] + 1)
            else:
                self.robot_move_strategy()
            self.bull_move()

            self.root.change_round_count()

    def robot_move_strategy(self):
        self.print_info("Robot change the position")
        dx = self.root.robot_position[0] - self.root.bull_position[0]
        dy = self.root.robot_position[1] - self.root.bull_position[1]
        if self.is_in_5X5(self.root.bull_position[0], self.root.bull_position[1], self.root.robot_position[0], self.root.robot_position[1]):
            # move to position(4, 5)
            dx1 = self.target_position[0] - self.root.robot_position[0]
            dy1 = self.target_position[1] - self.root.robot_position[1]
            t = False
            if abs(dx1) > 0:
                if dx1 > 0:
                    new_r_p = (self.root.robot_position[0] + 1, self.root.robot_position[1])
                else:
                    new_r_p = (self.root.robot_position[0] - 1, self.root.robot_position[1])
                if self.is_in_5X5(self.root.bull_position[0], self.root.bull_position[1], new_r_p[0], new_r_p[1]) and \
                        not self.collide_with_obstacles(new_r_p[0], new_r_p[1]) and new_r_p != self.root.bull_position:
                    self.root.robot_move(new_r_p[0], new_r_p[1])
                    t = True
            if abs(dy1) > 0:
                if dy1 > 0:
                    new_r_p = (self.root.robot_position[0], self.root.robot_position[1] + 1)
                else:
                    new_r_p = (self.root.robot_position[0], self.root.robot_position[1] - 1)
                if self.is_in_5X5(self.root.bull_position[0], self.root.bull_position[1], new_r_p[0], new_r_p[1]) and \
                        not self.collide_with_obstacles(new_r_p[0], new_r_p[1]) and new_r_p != self.root.bull_position:
                    self.root.robot_move(new_r_p[0], new_r_p[1])
                    t = True

            if abs(dx1) == 0 and abs(dy1) == 0:
                if self.root.bull_position == (self.root.robot_position[0]-1, self.root.robot_position[1]):
                    self.root.robot_move(self.root.robot_position[0]+1, self.root.robot_position[1])
                    t = True
                    self.target_yes = True
                elif self.root.bull_position == (self.root.robot_position[0]+1, self.root.robot_position[1]):
                    self.root.robot_move(self.root.robot_position[0] - 1, self.root.robot_position[1])
                    self.target_yes1 = True
                    t = True
            # if not t:
            #     new_r_p = self.get_random_robot_p()
            #     while new_r_p == self.root.bull_position and self.collide_with_obstacles(new_r_p[0], new_r_p[1]):
            #         new_r_p = self.get_random_robot_p()
            #     self.root.robot_move(new_r_p[0], new_r_p[1])

        else:
            if abs(dx) >= 0:
                if dx > 0:
                    new_r_p = (self.root.robot_position[0] - 1, self.root.robot_position[1])
                else:
                    new_r_p = (self.root.robot_position[0] + 1, self.root.robot_position[1])
                if not self.collide_with_obstacles(new_r_p[0], new_r_p[1]):
                    self.root.robot_move(new_r_p[0], new_r_p[1])
            if abs(dy) >= 0:
                if dy > 0:
                    new_r_p = (self.root.robot_position[0], self.root.robot_position[1] - 1)
                else:
                    new_r_p = (self.root.robot_position[0], self.root.robot_position[1] + 1)
                if not self.collide_with_obstacles(new_r_p[0], new_r_p[1]):
                    self.root.robot_move(new_r_p[0], new_r_p[1])

    def get_random_robot_p(self):
        random_t = self.random.randint(1, 8)
        if random_t == 1:
            return self.root.robot_position[0] - 1, self.root.robot_position[0]
        elif random_t == 2:
            return self.root.robot_position[0] + 1, self.root.robot_position[0]
        elif random_t == 3:
            return self.root.robot_position[0] - 1, self.root.robot_position[0] - 1
        elif random_t == 4:
            return self.root.robot_position[0] - 1, self.root.robot_position[0] + 1
        elif random_t == 5:
            return self.root.robot_position[0] + 1, self.root.robot_position[0] - 1
        elif random_t == 6:
            return self.root.robot_position[0] + 1, self.root.robot_position[0] + 1
        elif random_t == 7:
            return self.root.robot_position[0], self.root.robot_position[0] - 1
        elif random_t == 8:
            return self.root.robot_position[0], self.root.robot_position[0] + 1

    def bull_move(self):
        self.print_info("Bull change the position")
        self.print_info("Bull's current position " + str(self.root.robot_position))

        if self.is_in_5X5(self.root.bull_position[0], self.root.bull_position[1],
                          self.root.robot_position[0], self.root.robot_position[1]):
            dx = self.root.bull_position[0] - self.root.robot_position[0]
            dy = self.root.bull_position[1] - self.root.robot_position[1]
            temp_new_r_p = []
            if abs(dx) - abs(dy) != 0:
                if dx > 0:
                    new_r_p = (self.root.bull_position[0] - 1, self.root.bull_position[1])
                elif dx < 0:
                    new_r_p = (self.root.bull_position[0] + 1, self.root.bull_position[1])
                else:
                    new_r_p = (self.root.bull_position[0], self.root.bull_position[1])
                if not self.collide_with_obstacles(new_r_p[0], new_r_p[1]) and new_r_p != self.root.robot_position:
                    temp_new_r_p.append(new_r_p)

                if dy > 0:
                    new_r_p = (self.root.bull_position[0], self.root.bull_position[1] - 1)
                elif dy < 0:
                    new_r_p = (self.root.bull_position[0], self.root.bull_position[1] + 1)
                else:
                    new_r_p = (self.root.bull_position[0], self.root.bull_position[1])
                if not self.collide_with_obstacles(new_r_p[0], new_r_p[1]) and new_r_p != self.root.robot_position:
                    temp_new_r_p.append(new_r_p)
            elif abs(dx) - abs(dy) == 0:
                loguru.logger.info("Manhattan Distance = 1")
                direction = random.randint(0, 1)
                valid_p = []
                if direction:
                    if dx > 0:
                        new_r_p = (self.root.bull_position[0] - 1, self.root.bull_position[1])
                    elif dx < 0:
                        new_r_p = (self.root.bull_position[0] + 1, self.root.bull_position[1])
                    else:
                        new_r_p = (self.root.bull_position[0], self.root.bull_position[1])
                    if not self.collide_with_obstacles(new_r_p[0], new_r_p[1]) and new_r_p != self.root.robot_position:
                        valid_p.append(new_r_p)
                else:
                    if dy > 0:
                        new_r_p = (self.root.bull_position[0], self.root.bull_position[1] - 1)
                    elif dy < 0:
                        new_r_p = (self.root.bull_position[0], self.root.bull_position[1] + 1)
                    else:
                        new_r_p = (self.root.bull_position[0], self.root.bull_position[1])
                    if not self.collide_with_obstacles(new_r_p[0], new_r_p[1]) and new_r_p != self.root.robot_position:
                        valid_p.append(new_r_p)
                if len(valid_p) > 0:
                    min_p = valid_p[0]
                    min_d = abs(self.root.robot_position[0] - valid_p[0][0]) + abs(self.root.robot_position[1] - valid_p[0][1])
                    for i in range(len(valid_p)):
                        if abs(self.root.robot_position[0] - valid_p[i][0]) + abs(self.root.robot_position[1] - valid_p[i][1]) < min_d:
                            min_p = valid_p[i]
                            min_d = abs(self.root.robot_position[0] - valid_p[i][0]) + abs(self.root.robot_position[1] - valid_p[i][1])
                    self.root.bull_move(min_p[0], min_p[1])
            if len(temp_new_r_p) > 0:
                select_p = []
                min_p = temp_new_r_p[0]
                min_d = abs(self.root.robot_position[0] - temp_new_r_p[0][0]) + abs(
                    self.root.robot_position[1] - temp_new_r_p[0][1])
                for i in range(len(temp_new_r_p)):
                    if abs(self.root.robot_position[0] - temp_new_r_p[i][0]) + abs(
                            self.root.robot_position[1] - temp_new_r_p[i][1]) < min_d:
                        min_p = temp_new_r_p[i]
                        min_d = abs(self.root.robot_position[0] - temp_new_r_p[i][0]) + abs(
                            self.root.robot_position[1] - temp_new_r_p[i][1])
                    elif abs(self.root.robot_position[0] - temp_new_r_p[i][0]) + abs(
                            self.root.robot_position[1] - temp_new_r_p[i][1]) == min_d:
                        if temp_new_r_p[i] not in select_p:
                            select_p.append(temp_new_r_p[i])
                p = select_p[random.randint(0, len(select_p)-1)]

                if p == self.root.X:
                    a = messagebox.showinfo("Warning", 'Game Over.The bull walks onto the x')
                    print("Error code ", a)
                    self.continue_game = False
                else:
                    self.root.bull_move(p[0], p[1])
        else:
            self.move1()

    def charge_is_valid(self, d):
        if d[0] > 0:
            if d[1] > 0:
                for i in range(1, d[1]):
                    if (self.root.bull_position[0], self.root.bull_position[1] + i) in self.root.walls:
                        return False
            else:
                for i in range(1, d[1]):
                    if (self.root.bull_position[0], self.root.bull_position[1] - i) in self.root.walls:
                        return False
        else:
            if d[1] > 0:
                for i in range(1, d[1]):
                    if (self.root.bull_position[0] + i, self.root.bull_position[1]) in self.root.walls:
                        return False
            else:
                for i in range(1, d[1]):
                    if (self.root.bull_position[0] - i, self.root.bull_position[1]) in self.root.walls:
                        return False

    def get_min_distance(self, pos1, pos2):
        d1 = pos2[0] - pos1[0]
        d2 = pos2[1] - pos1[1]
        if d1 == 0 or d2 == 0:
            if abs(d1) - abs(d2) > 0:
                return 0, -1, d2
            else:
                return 0, 1, d1
        else:
            if abs(d1) - abs(d2) > 0:
                return -1, d2
            else:
                return 1, d1

    def move1(self):
        direction = self.random.randint(1, 4)
        new_position = self.generate_new_direction(direction, self.root.bull_position[0], self.root.bull_position[1])
        full = [direction]
        new_p = True
        while (not self.position_is_valid(new_position[0], new_position[1])) and \
                self.collide_with_obstacles(new_position[0], new_position[1]):
            if direction not in full:
                full.append(direction)

            if len(full) == 4:
                self.print_info("Shy bull.")
                new_p = False
                break
            direction = self.random.randint(1, 4)
            new_position = self.generate_new_direction(direction, self.root.bull_position[0],
                                                       self.root.bull_position[1])
        if new_p:
            self.print_info("Bull's new position " + str(new_position))

        if new_position == self.root.X:
            a = messagebox.showinfo('Game Over.The bull walks onto the x')
            print("Error code ", a)
        else:
            self.root.bull_move(new_position[0], new_position[1])

    def generate_new_direction(self, d, pos_x, pos_y):
        if d == 1:
            pos_x -= 1
        elif d == 2:
            pos_x += 1
        elif d == 3:
            pos_y -= 1
        elif d == 4:
            pos_y += 1
        elif d == 5:
            pos_x -= 1
            pos_y += 1
        elif d == 6:
            pos_x -= 1
            pos_y -= 1
        elif d == 7:
            pos_x += 1
            pos_y -= 1
        elif d == 8:
            pos_x += 1
            pos_y += 1
        return pos_x, pos_y

    def print_info(self, info):
        pass
        # loguru.logger.info(info)

    def position_is_valid(self, pos_x, pos_y):
        if self.root.walls[1] >= pos_x >= self.root.walls[0] and \
                self.root.walls[1] >= pos_y >= self.root.walls[0]:
            return True
        else:
            return False

    def collide_with_obstacles(self, pos_x, pos_y):
        if (pos_x, pos_y) in self.root.obstacles:
            self.print_info("collide with obstacles, " + str((pos_x, pos_y)))
            return True
        else:
            if self.position_is_valid(pos_x, pos_y):
                return False
            else:
                return True

    def is_in_5X5(self, b_pos_x, b_pos_y, r_pos_x, r_pos_y):
        if -3 < r_pos_x - b_pos_x < 3 and -3 < r_pos_y - b_pos_y < 3:
            return True
        else:
            return False


if __name__ == '__main__':
    root = None
    m = myThread(root)
    print(m.is_in_5X5(0, 0, -2, -2))
