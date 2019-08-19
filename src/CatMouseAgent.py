import numpy as np
from math import pi, ceil

'''

The agent that handles the mechanisms/etc of the Cat and Mouse environment.

'''



class CatMouseAgent:


    def __init__(self, **kwargs):

        self.N_actions = 2


        self.circle_rad = 1.0


        self.cat_angle = 0.0 # goes from 0 to 2*pi, gets normalized to -1 to 1
        self.mouse_pos = np.array([0.0, 0.0])
        self.iter_counter = 0

        self.mouse_speed_rad_div = 5.0
        self.mouse_speed = self.circle_rad/self.mouse_speed_rad_div

        self.set_cat_speed(kwargs.get('cat_speed_rel', 3.0))

        #self.max_ep_steps = ceil(5*self.mouse_speed_rad_div)
        self.max_ep_steps = kwargs.get('max_ep_steps', 200)

        self.escape_reward = 1.0
        self.caught_penalty = -0.5
        self.time_penalty = -1.0/self.max_ep_steps


    def set_cat_speed(self, cat_speed_rel):
        self.cat_speed_rel = cat_speed_rel

        self.cat_speed = self.cat_speed_rel*self.mouse_speed
        circle_circum = 2*pi*self.circle_rad

        # This is the max angle diff the cat can go in an iteration
        self.cat_angle_d = 2*pi*self.cat_speed/circle_circum


    def getStateVec(self):
        #return(np.concatenate(([self.iter_counter], self.mouse_pos, [self.cat_angle/pi - 1])))
        #return(np.concatenate(([0.0], self.mouse_pos, [self.cat_angle/pi - 1])))
        return(np.concatenate((self.mouse_pos, [np.cos(self.cat_angle), np.sin(self.cat_angle)])))


    def initEpisode(self):
        self.cat_angle = 2*pi*np.random.uniform() # goes from 0 to 2*pi, gets normalized to -1 to 1
        self.mouse_pos = np.array([0.0, 0.0])
        self.iter_counter = 0


    def iterate(self, a):

        dx, dy = a
        move_ang = np.arctan2(dy, dx)

        self.mouse_pos += self.mouse_speed*np.array([np.cos(move_ang), np.sin(move_ang)])

        # This is the angle of the mouse from center of the circle.
        mouse_angle = np.arctan2(self.mouse_pos[1], self.mouse_pos[0])%(2*pi)
        done = False

        if np.linalg.norm(self.mouse_pos) >= self.circle_rad:
            # Meaning the mouse got out...
            done = True

            angle_diff = np.abs(self.cat_angle - mouse_angle)

            # Check if where the mouse got out is in the cat's angular
            # range that it can get to
            if min(angle_diff, 2*pi - angle_diff) < self.cat_angle_d:
                # This means the cat caught it!
                #print('Mouse caught!')
                reward = self.caught_penalty
                self.cat_angle = mouse_angle
            else:
                #print('Mouse escaped!')
                reward = self.escape_reward

        else:
            # This is if you're still in the circle but haven't escaped.
            reward = self.time_penalty

            ## Wait, this needs to be fixed: if it hasn't escaped but the cat can be as close
            # as possible, it's not making it do that

            cat_mouse_angle_diff = np.abs(self.cat_angle - mouse_angle)

            if min(cat_mouse_angle_diff, 2*pi - cat_mouse_angle_diff) <= self.cat_angle_d:
                self.cat_angle = mouse_angle

            else:
                # This makes the cat go in the direction that gets it as close to the mouse as possible.
                angle_diff_L = np.abs((self.cat_angle - self.cat_angle_d) - mouse_angle)
                angle_diff_L = min(angle_diff_L, 2*pi - angle_diff_L)

                angle_diff_R = np.abs((self.cat_angle + self.cat_angle_d) - mouse_angle)
                angle_diff_R = min(angle_diff_R, 2*pi - angle_diff_R)

                if angle_diff_L < angle_diff_R:
                    self.cat_angle = (self.cat_angle - self.cat_angle_d)%(2*pi)
                else:
                    self.cat_angle = (self.cat_angle + self.cat_angle_d)%(2*pi)

        if self.iter_counter >= self.max_ep_steps:
            #done = True
            pass

        sv = self.getStateVec()
        self.iter_counter += 1
        return(reward, sv, done)











#
