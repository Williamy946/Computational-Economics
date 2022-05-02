import agent
import env
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from agent import Agent
from agent import ReplayMemory
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class DynamicPricing():
    def __init__(self, args, datasets):

        # Raw Datasets
        self.region_hotel, self.region_price, \
        self.region_score, self.region_accommodate, \
        self.guest_num, self.guest_mean_price, \
        self.guest_std_price = tuple(datasets)
        self.dynamic_num = args.num_dynamic_agent
        self.region = args.region  # len(self.region_hotel) # index of region

        # Env


        self.n_hotel = [len(hotels) for hotels in self.region_hotel] 
        self.user_dist = self.guest_num
        self.num_steps = args.max_step
        self.hotel_ids = self.region_hotel
        self.hotel_price = self.region_price
        self.hotel_score = self.region_score
        self.hotel_accommodate = self.region_accommodate
        self.user_exp_mean = self.guest_mean_price
        self.user_exp_std = self.guest_std_price
        self.review_times = args.review_times

        # DQN
        self.tau = args.tau
        self.gamma = args.gamma
        self.p_lr = args.p_lr
        self.p_lr2 = args.p_lr2
        self.v_lr = args.v_lr
        self.v_lr2 = args.v_lr2
        self.memory_size = args.memory_size
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.double_q = args.double_q

        # train
        self.train_step = args.train_times
        self.sample_step = args.sample_times
        self.batch_size = args.batch_size
        self.update_times = args.update
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:" + str(args.device)) if self.use_cuda else torch.device("cpu")

        #self.env = env.Env(self.n_hotel, self.n_region, self.num_steps,
        #                   self.hotel_ids,self.hotel_score, self.hotel_accommodate,
        #                   self.hotel_price, self.user_dist, self.user_exp_mean, self.user_exp_std)
        #self.env.reset()


    def train(self):
        #dynamic_hotel = np.random.randint(0, self.n_hotel, 20)#np.array([0])

        region_n_hotel = self.n_hotel[self.region]
        region_user_dist = self.guest_num[self.region]
        region_user_dist = region_user_dist + [1]*11
        region_hotel_ids = self.region_hotel[self.region]
        region_hotel_price = self.region_price[self.region]
        region_hotel_score = self.region_score[self.region]
        region_hotel_accommodate = self.region_accommodate[self.region]
        region_user_exp_mean = self.guest_mean_price[self.region]
        region_user_exp_std = self.guest_std_price[self.region]
        hotel_id = [i for i in range(region_n_hotel)]

        dynamic_hotel = np.random.choice(hotel_id, self.dynamic_num,replace=False)
        #dynamic_hotel[0] = 96
        self.env = env.Env(region_n_hotel, self.region, self.num_steps,
                           region_hotel_ids, region_hotel_score, region_hotel_accommodate,
                           region_hotel_price, region_user_dist, region_user_exp_mean, region_user_exp_std, self.review_times)
        memory = ReplayMemory(self.memory_size)

        agent_list = []
        for dyn in range(len(dynamic_hotel)):
            agent_list.append(Agent(np.array(dyn).reshape(1), self.device, region_n_hotel, self.p_lr, self.p_lr2, self.v_lr, self.v_lr2, self.memory_size,
                    self.batch_size, self.eps_start, self.eps_end, self.eps_decay, self.gamma, self.tau, self.double_q, memory))



        for i in range(self.train_step):#
            print("Step: " + str(i))
            mean_gain = 0.0
            mean_diff = 0.0
            for itr in tqdm(range(self.sample_step),desc='sampling'):
                cur_state, done = self.env.reset()
                accum_reward = np.array([])
                accum_diff = 0.0
                abs_accu_re = 0.0
                action_series = np.array([])
                profit_series = np.array([])
                fit_profit_series = np.array([])
                step = 1
                while not done:
                    action_ser = np.array([])
                    for age in agent_list:
                        action, val = age.choose_action(cur_state, step)
                        action_ser = np.append(action_ser, action)

                    next_state, self_state, r, profit, fit_profit, done = self.env.step(action_ser.reshape(1, len(dynamic_hotel)), dynamic_hotel, step)
                    #accum_reward += r.sum()
                    abs_accu_re += np.abs(r)
                    #accum_diff += np.abs(val.cpu().detach()-r)
                    memory.push(torch.tensor(cur_state), torch.tensor(action_ser), torch.tensor(next_state), torch.tensor(r))
                    cur_state = next_state#np.concatenate((cur_state,next_state),axis=1)
                    if step == 1:
                        accum_reward = r.reshape(len(dynamic_hotel), 1)
                        action_series = action_ser.reshape(len(dynamic_hotel), 1)
                        profit_series = profit.reshape(len(dynamic_hotel), 1)
                        fit_profit_series = fit_profit.reshape(len(dynamic_hotel), 1)
                    else:
                        accum_reward = np.append(accum_reward, r.reshape(len(dynamic_hotel), 1), axis=1)
                        action_series = np.append(action_series, action_ser.reshape(len(dynamic_hotel), 1), axis=1)
                        profit_series = np.append(profit_series, profit.reshape(len(dynamic_hotel), 1), axis=1)
                        fit_profit_series = np.append(fit_profit_series, fit_profit.reshape(len(dynamic_hotel), 1), axis=1)
                    step += 1

                mean_gain += accum_reward/(self.num_steps)
                #mean_diff += accum_diff[0][0].cpu().detach().numpy()/(abs_accu_re*self.sample_step)
                print()
                print("Mean dynamic gain:" + str(np.sum(accum_reward, axis=1)/self.num_steps))
                print("Profit Series: " + str(np.sum(profit_series, axis=1)/np.sum(fit_profit_series, axis=1)))
                print("Profit Mean: " + str(np.mean(np.sum(profit_series, axis=1) / np.sum(fit_profit_series, axis=1))))
                print(action_series)
                #print("Mean dynamic diff:" + str(accum_diff[0][0].cpu().detach().numpy()/(abs_accu_re)))
            print()
            print("Mean dynamic gain:" + str(np.sum(mean_gain, axis=1)/self.sample_step))
            print("Mean dynamic diff:" + str(mean_diff))

            mean_policy = 0.0
            mean_value = 0.0
            mean_learn_diff = 0.0
            num = 0
            for itr in tqdm(range(self.update_times),desc='updating'):
                for age in agent_list:
                    policy, value, diff = age.learn()
                    mean_policy += policy/(self.update_times*self.dynamic_num)
                    mean_value += value/(self.update_times*self.dynamic_num)
                    mean_learn_diff += diff/(self.update_times*self.dynamic_num)

            print()
            print("Policy Loss: " + str(mean_policy))
            print("Value Loss: " + str(mean_value))
            print("Reward mean diff: " + str(mean_learn_diff))


        cur_state, done = self.env.reset()
        accum_reward = np.array([])
        accum_diff = 0.0
        step = 1
        action_series = np.array([])
        profit_series = np.array([])
        fit_profit_series = np.array([])
        while not done:
            action_ser = np.array([])
            #profit_ser = np.array([])
            for age in agent_list:
                action, val = age.choose_action(cur_state, step, True)
                action_ser = np.append(action_ser, action)

            next_state, self_state, r, profit, fit_profit, done = self.env.step(action_ser, dynamic_hotel, step)
            #accum_reward += r.sum()
            #accum_diff += np.abs(val.cpu().detach() - r)
            memory.push(torch.tensor(cur_state), torch.tensor(action_ser), torch.tensor(next_state), torch.tensor(r))
            cur_state = next_state#np.concatenate((cur_state, next_state), axis=1)
            if step == 1:
                accum_reward = r.reshape(len(dynamic_hotel), 1)
                action_series = action_ser.reshape(len(dynamic_hotel),1)
                profit_series = profit.reshape(len(dynamic_hotel),1)
                fit_profit_series = fit_profit.reshape(len(dynamic_hotel), 1)
            else:
                accum_reward = np.append(accum_reward, r.reshape(len(dynamic_hotel), 1), axis=1)
                action_series = np.append(action_series, action_ser.reshape(len(dynamic_hotel),1), axis=1)
                profit_series = np.append(profit_series, profit.reshape(len(dynamic_hotel),1), axis=1)
                fit_profit_series = np.append(fit_profit_series, fit_profit.reshape(len(dynamic_hotel), 1), axis=1)
            step += 1

        print()
        print("Mean dynamic improvement:" + str(np.sum(accum_reward, axis=1)/self.num_steps))
        print("Profit Series: " + str(np.sum(profit_series, axis=1)/np.sum(fit_profit_series, axis=1)))
        print("Profit Mean: " + str(np.mean(np.sum(profit_series, axis=1) / np.sum(fit_profit_series, axis=1))))
        #print("Mean dynamic diff:" + str(accum_diff[0][0].cpu().detach().numpy() / self.num_steps))

        # Plot for action curve

        sub_axix = np.array([i for i in range(self.num_steps)]).reshape(self.num_steps,1)
        profit_data = pd.DataFrame(np.append(sub_axix, profit_series.reshape(self.num_steps, -1),axis=1))
        fit_profit_data = pd.DataFrame(np.append(sub_axix, fit_profit_series.reshape(self.num_steps, -1), axis=1))
        #action_data = pd.DataFrame(np.append(sub_axix, action_series.reshape(self.num_steps, -1), axis=1))
        #for i in range(5):
            #plt.plot(sub_axix ,profit_series[i])
            #plt.plot(sub_axix, action_series[i])
        #for i in range(self.dynamic_num):
        #color = (sns.dark_palette("purple"))
        #sns.set_context("paper")
        #sns.lineplot(data=profit_series.reshape(self.num_steps, -1))
        #sns.lineplot(data=action_series.reshape(self.num_steps,-1))
        #plt.show()
        #sns.lineplot(data=profit_series.reshape(self.num_steps, -1)[::2,::2])
        #plt.show()
        print()
        sns.set_context("paper")
        # sns.lineplot(data=profit_series.reshape(self.num_steps, -1))
        maxind = np.argmax(np.sum(profit_series, axis=1) / np.sum(fit_profit_series, axis=1))
        minind = np.argmin(np.sum(profit_series, axis=1) / np.sum(fit_profit_series, axis=1))
        # maxind['region'] = 'Max'
        data = pd.DataFrame(action_series[[maxind, minind], :]).transpose()
        data.columns = ['Max', 'Min']
        sns.lineplot(data=data)
        # sns.lineplot(data=action_series[minind])
        plt.show()
        # sns.lineplot(data=profit_series.reshape(self.num_steps, -1)[::2,::2])
        # plt.show()
        print()