import math
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil
import os

# constant values
length = 630
width = 303
ball_d = 32
valid_hitting_angle = 105

hole_safe_r = 0.5*ball_d
targets = np.array([[ball_d/2, ball_d/2], [length/2, ball_d/2], [length - ball_d/2, ball_d/2],
                    [ball_d/2, width-ball_d/2], [length/2, width-ball_d/2], [length-ball_d/2, width-ball_d/2]])
holes = np.array([[0, 0], [length/2, 0], [length, 0],
                  [0, width], [length/2, width], [length, width]])

# global variable
cue_coor = np.array([0, 0])
nine_coor = np.array([0, 0])
obs_coor = np.array([[0, 0], [0, 0]])


# energy
total_energy = 1000
friction = 1
bounce_lost = 500
best_energy = -10000
best_direction = np.array([0, 0])

# prevent stick hitting obstacles
stick_path_width = 50
stick_path_length = 50


def random_coor():
    x = random.randrange(int(ball_d/2), int(length-ball_d/2))
    y = random.randrange(int(ball_d/2), int(width-ball_d/2))
    return [x, y]


def angle(start, end_a, end_b):
    a = start - end_a
    b = start - end_b
    cos = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))
    cos = np.clip(cos, -1, 1)  # 0 and 180 dot not equal to multiplication
    angle = np.degrees(np.arccos(cos))
    return angle


def blocked(start, end, obs, obs_r):
    valid_dis = obs_r + ball_d/2
    if np.linalg.norm(obs-start) <= valid_dis or np.linalg.norm(obs-end) <= valid_dis:
        return True

    dis = np.linalg.norm(np.cross(obs-start, end-start)) / \
        np.linalg.norm(end-start)
    ang_end = angle(end, obs, start)
    ang_start = angle(start, obs, end)
    if dis <= valid_dis and ang_start <= 90 and ang_end <= 90:
        return True

    return False


def obs_blocked(start, end):
    for obs in obs_coor:
        if blocked(start, end, obs, ball_d/2):
            return True
    return False


def hole_blocked(start, end):
    for hole in holes:
        if blocked(start, end, hole, hole_safe_r):
            return True
    return False


def away_from_holes(target):
    for hole in holes:
        dis = np.linalg.norm(hole - target)
        if dis <= hole_safe_r:
            return False
    return True


def plot():
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([0, length])
    plt.ylim([0, width])
    cue_circle = plt.Circle(cue_coor, ball_d/2, color=(1, 0, 0), zorder=10)
    nine_circle = plt.Circle(nine_coor, ball_d/2, color=(1, 1, 0), zorder=10)
    for obs in obs_coor:
        obs_circle = plt.Circle(obs, ball_d/2, color=(0, 0, 0), zorder=10)
        plt.gcf().gca().add_artist(obs_circle)
    for hole in holes:
        hole_circle = plt.Circle(hole, ball_d, color=(0, 0, 0), zorder=10)
        plt.gcf().gca().add_artist(hole_circle)
    plt.gcf().gca().add_artist(cue_circle)
    plt.gcf().gca().add_artist(nine_circle)

    # if random_test:
    #     filename = 'plots/' + str(num1) + '.png'
    #     plt.savefig(filename)
    # else:
    plt.show()
    plt.clf()


def setup_points(cue_coordinates, nine_ball_coordinates, obstacle_coordinates_1, obstacle_coordinates_2):
    global cue_coor, nine_coor, obs_coor
    # input
    # if not random_test:
    #     cue_coor = np.asarray([int(x)
    #                            for x in input("Cue Ball x,y :").split(',')])
    #     nine_coor = np.asarray([int(x)
    #                             for x in input("Nine Ball x,y :").split(',')])
    #     obs_coor = np.array([[0, 0], [0, 0]])
    #     obs_coor[0] = np.asarray([int(x)
    #                               for x in input("Fisrt Obstacle Ball x,y :").split(',')])
    #     obs_coor[1] = np.asarray(
    #         [int(x) for x in input("Second Obstacle Ball x,y :").split(',')])
    # else:
    #     cue_coor = np.array(random_coor())
    #     nine_coor = np.array(random_coor())
    #     obs_coor = np.array([[0, 0], [0, 0]])
    #     obs_coor[0] = np.array(random_coor())
    #     obs_coor[1] = np.array(random_coor())

    cue_coor = np.array(cue_coordinates)
    nine_coor = np.array(nine_ball_coordinates)
    obs_coor[0] = np.array(obstacle_coordinates_1)
    obs_coor[1] = np.array(obstacle_coordinates_2)

    print("Cue Coordinate :", cue_coor, " Nine Coordinate : ", nine_coor)
    print("Obstacle 1 :", obs_coor[0], " Obstacle 2 : ", obs_coor[1])


def bounce(start, end, wall, xy):
    a = np.abs(start[xy] - wall[xy])
    b = np.abs(end[xy] - wall[xy])
    # reverse xy
    yx = 1-xy
    if xy:
        return np.array([(start[yx] * b + end[yx] * a)/(a+b), wall[xy]])
    else:
        return np.array([wall[xy], (start[yx] * b + end[yx] * a)/(a+b)])


def distance(start, end):
    delta = end-start
    dis = np.linalg.norm(delta)
    return dis


def plot_path(start, end, color):
    # plot
    plt.plot([start[0], end[0]], [start[1], end[1]], color=color)
    circle = plt.Circle(end, ball_d/2, color=color)
    plt.gcf().gca().add_artist(circle)


def plot_answer(start, direction, color):
    # plot
    plt.arrow(start[0], start[1], direction[0], direction[1], color=color, length_includes_head=True,
              head_width=15, head_length=15)


def stick_hits_obstacles(cue, target):
    if distance(nine_coor, cue_coor) < stick_path_length and angle(cue, target, nine_coor) > 90:
        dis = np.linalg.norm(np.cross(target-cue, cue-nine_coor)
                             )/np.linalg.norm(target-cue)
        if dis < stick_path_width/2:
            return True
    for obs in obs_coor:
        if distance(obs, cue_coor) < stick_path_length and angle(cue, target, obs) > 90:
            dis = np.linalg.norm(np.cross(target-cue, cue-obs)
                                 )/np.linalg.norm(target-cue)
            if dis < stick_path_width/2:
                return True
    return False


def get_direction(target):
    has_solve = False
    global best_energy, best_direction

    # first check if the path between nine ball and hole is blocked
    if obs_blocked(nine_coor, target):
        print("    Blocked between nine and target")
        return False

    # then check if ghost ball position is valid
    delta = nine_coor - target
    ghost_coor = nine_coor + delta/np.linalg.norm(delta)*ball_d
    if not (ball_d/2 <= ghost_coor[0] <= (length - ball_d/2) and ball_d/2 <= ghost_coor[1] <= (width - ball_d/2)):
        print("    Ghost position out of bound")
        return False

    if not away_from_holes(ghost_coor):
        print("    Ghost position on holes' position")
        return False
    # color = (random.random(), random.random(), random.random(), 0.3)
    print("    Direct hit : ")
    # then check if the hitting angle is valid
    # or if paths before hitting has been blocked
    # or if paths before hitting has been blocked by holes
    ang = angle(ghost_coor, cue_coor, nine_coor)
    if ang <= valid_hitting_angle:
        print("        Invalid hitting angle")
    elif obs_blocked(cue_coor, ghost_coor):
        print("        Blocked between cue and ghost")
    elif hole_blocked(cue_coor, ghost_coor):
        print("        Cue path over holes' position")
    elif stick_hits_obstacles(cue_coor, ghost_coor):
        print("        Stick hits obstacles")
    else:

        has_solve = True

        # calculate energy
        cue_distance = distance(cue_coor, ghost_coor)
        nine_distance = distance(nine_coor, target)
        vert = np.abs(np.cos(np.radians(ang)))
        bounce_count = 0
        energy = (total_energy - cue_distance*friction -
                  bounce_count*bounce_lost)*vert - nine_distance*friction
        if energy <= 0:
            print("        Can't reach destination")
            color = (1, 0, 0, 0.3)
        else:
            color = (0, 0, 0, energy/total_energy)

        # calculate direction
        direction = (ghost_coor - cue_coor) / \
            np.linalg.norm(ghost_coor - cue_coor)
        if energy > best_energy:
            best_energy = energy
            best_direction = direction

        # plot
        plot_path(cue_coor, ghost_coor, color)
        plot_path(nine_coor, target, color)

        # print
        print("        Hitting direction : ", direction)
        print("        Energy : ", energy)

    # try with one bounce
    # for convenience, use the first four hole position to represent 4 side of walls
    # take x when i%2 = 0, take y when i%2 = 1
    # ["ball_d/2", ball_d/2], [length/2, "ball_d/2"], ["length - ball_d/2", ball_d/2],[ball_d/2, "width-ball_d/2"]
    for i in range(4):
        # check if cue ball is too close to the wall
        if np.abs(cue_coor[i % 2] - (targets[i])[i % 2]) < 50:
            continue
        # get bounce position
        bounce_coor = bounce(cue_coor, ghost_coor, targets[i], i % 2)
        # check if the stick hits anything
        if stick_hits_obstacles(cue_coor, bounce_coor):
            continue
        # check if bounce position is blocked by holes
        if hole_blocked(cue_coor, bounce_coor):
            continue
        # check if path before bounce is blocked by obstacle
        if obs_blocked(cue_coor, bounce_coor):
            continue
        # check if path is blocked by nine ball
        if blocked(cue_coor, bounce_coor, nine_coor, ball_d/2):
            continue
        # check if hitting angle is valid
        ang = angle(ghost_coor, bounce_coor, nine_coor)
        if ang <= valid_hitting_angle:
            continue
        # check if path after bounce is blocked
        if obs_blocked(bounce_coor, ghost_coor):
            continue
        # check if bounce to ghost is blocked by holes
        if hole_blocked(bounce_coor, ghost_coor):
            continue

        has_solve = True
        print("    One bounce dir", i, ": ")
        # energy
        cue_distance = distance(cue_coor, bounce_coor) + \
            distance(bounce_coor, ghost_coor)
        nine_distance = distance(nine_coor, target)
        vert = np.abs(np.cos(np.radians(ang)))
        bounce_count = 1
        energy = (total_energy - cue_distance*friction -
                  bounce_count*bounce_lost)*vert - nine_distance*friction
        if energy <= 0:
            print("        Can't reach destination")
            color = (1, 0, 0, 0.3)
        else:
            color = (0, 0, 0, energy/total_energy)

        # calculate direction
        direction = (bounce_coor - cue_coor) / \
            np.linalg.norm(bounce_coor - cue_coor)
        if energy > best_energy:
            best_energy = energy
            best_direction = direction

        # plot
        plot_path(cue_coor, bounce_coor, color)
        plot_path(bounce_coor, ghost_coor, color)
        plot_path(nine_coor, target, color)

        # print

        print("        Hitting direction : ", direction)
        print("        Energy : ", energy)

    if not has_solve:
        print("    No result")

    return has_solve


# if random_test:
#     if os.path.exists("plots"):
#         shutil.rmtree("plots")
#     os.mkdir("plots")
#     for i in range(tests):
#         setup_points()
#         plt.figure()
#         for target in targets:
#             print("Target Coordinate : ", target)
#             get_direction(target)
#         plot(i)
# else:
#     setup_points()
#     for target in targets:
#         print("Target Coordinate : ", target)
#         get_direction(target)
#     plot(0)

def result():
    solved = False
    for target in targets:
        print("\nTarget Coordinate : ", target)
        if get_direction(target):
            solved = True
    if not solved:
        global best_direction
        best_direction = (nine_coor - cue_coor) / \
            np.linalg.norm(nine_coor - cue_coor)

        # check if stick is blocked
        if stick_hits_obstacles(cue_coor, nine_coor):
            print("hi")
            delta_x = cue_coor[0]-nine_coor[0]
            delta_y = cue_coor[1]-nine_coor[1]
            for i in range(max(abs(delta_x), abs(delta_y))*2):
                temp_x_coor = nine_coor + [i*delta_x/abs(delta_x), 0]
                if not stick_hits_obstacles(cue_coor, temp_x_coor):
                    best_direction = (temp_x_coor - cue_coor) / \
                        np.linalg.norm(temp_x_coor - cue_coor)
                    break

                temp_y_coor = nine_coor + [0, i*delta_y/abs(delta_y)]
                if not stick_hits_obstacles(cue_coor, temp_y_coor):
                    best_direction = (temp_y_coor - cue_coor) / \
                        np.linalg.norm(temp_y_coor - cue_coor)
                    break

    plot_answer(cue_coor, best_direction*100, (0, 1, 0, 1))
    return best_direction
