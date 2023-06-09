from gym_multigrid.multigrid import *
import time

class CollectGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to collect the balls
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        num_balls=[],
        agents_index = [],
        balls_index=[],
        balls_reward=[],
        zero_sum = False,
        view_size=7,
        ps = True

    ):
    
        self.num_balls = num_balls
        self.balls_index = balls_index
        self.balls_reward = balls_reward
        self.zero_sum = zero_sum
        self.donedone = False
        self.remaining_ball = 2
        self.world = World
        self.things = []
      

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))
            #print("hi")

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size,
            partial_obs = ps
        )



    def _gen_grid(self, width, height):
        #grid object generated here
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)


        counter1 = 0
        for index, reward in zip(self.balls_index, self.balls_reward):
            
            #print(index)
            if counter1 == 0:
                aball = Ball(self.world, index, reward)
                self.place_obj(aball,thepos=[5,5])
            else:
                if counter1 == 1:
                    aball = Ball(self.world, index, reward)
                    self.place_obj(aball,thepos=[1,5])
                else:
                    aball = Ball(self.world, index, reward)
                    self.place_obj(aball,thepos=[5,1])
            
            self.things.append(aball)
            counter1 +=1
            

        # Randomize the player start position and orientation
        counter2 = 0
        for a in self.agents:
            if counter2 == 0:
                self.place_agent(a)
            else:
                if counter2 == 1:
                    self.place_agent(a)
                else:
                    self.place_agent(a)
            self.things.append(a)
            counter2 += 1
    def reset(self):
        self.things = []
        obs = super().reset()
        self.donedone = False
        self.remaining_ball = 2
      
        info = [thing.cur_pos for thing in self.things[0:3]]
        info += [thing.pos for thing in self.things[3:6]]
        info += [thing.dir for thing in self.things[3:6]]
        if info[1][0] == -1:
            info.append(0)
        else:
            info.append(1)
        if info[2][0] == -1:
            info.append(0)
        else:
            info.append(1)
        
        
        return info

    def _reward(self, i, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        for j,a in enumerate(self.agents):
            if a.index==i:
                rewards[j]+=reward        
                #print(f'team {a.index} rewarded')
            if self.zero_sum:
                if a.index!=i:
                    rewards[j] -= reward
                    #print(f'team {a.index} punished')

    def _handle_pickup(self, i, rewards, cur_pos, cur_cell): ##mode this function to disallow picking up opposing team's flag
        #print("efmfelfm")
        if cur_cell:
            #print("djjdnc")
            if cur_cell.type == "box":
                #print("enenkded")
                #if fwd_cell.index in [0, self.agents[i].index]:
                if cur_cell.get_ball().index == self.agents[i].index and cur_cell.get_ball().type == "ball":
                    # if the ball in the current cell has the same index
                    #print("picking up")
                    cur_cell.get_ball().cur_pos = np.array([-1, -1])
                    #time.sleep(5)
                    cur_cell.remove_ball()
                    self.grid.set(*cur_pos, cur_cell.get_agent())
                    team = self.agents[i].index
                    #print(f'team {team} pick up')
                    if self.agents[i].index == 2:
                        self.remaining_ball -= 1  
                        #print(f'self.remaining_ball {self.remaining_ball}')          
                        if self.remaining_ball <= 0 :
                            self._reward(team, rewards, 20)
                            self.donedone = True
                        if self.remaining_ball == 1:
                            self._reward(team, rewards, 20)
                        
                    else:
                        self._reward(team, rewards, 40)
                        self.donedone = True
                else:
                    kkk = "good"
                    #print(f'agent {i} illegal pick up')

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def step(self, actions):
        
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        if self.donedone:
            done = True
        
        return obs, rewards, done, info


class CollectGame4HEnv10x10N2(CollectGameEnv):
    def __init__(self):
        super().__init__(size=10,
        num_balls=[3],
        agents_index = [1,2,2],
        balls_index=[1,2,2],
        balls_reward=[1,1,1],
        zero_sum=True)
class CollectGamefullobs(CollectGameEnv):
    def __init__(self):
        super().__init__(size=10,
        num_balls=[3],
        agents_index = [1,2,2],
        balls_index=[1,2,2],
        balls_reward=[1,1,1],
         
        zero_sum=True,ps=False)
class CollectGame5by5(CollectGameEnv):
    def __init__(self):
        super().__init__(size=7,
        num_balls=[3],
        agents_index = [1,2,2],
        balls_index=[1,2,2],
        balls_reward=[1,1,1],
         
        zero_sum=True,ps=False)
       


