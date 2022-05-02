from tqdm import tqdm
import numpy as np
import torch
import os
import copy

class Env():
    def __init__(self, n_hotel, n_region,  num_steps, hotel_ids, hotel_score, hotel_accommodate, hotel_price,
                 user_dist, user_exp_mean, user_exp_std, review_times):
        self.num_steps = num_steps

        # region property
        self.n_hotel = n_hotel
        self.n_region = n_region # index_of_region

        # hotel property
        self.hotel_ids = hotel_ids
        self.hotel_price = np.array(hotel_price)
        self.init_price = np.array(hotel_price)
        self.hotel_score = np.array(hotel_score)
        self.hotel_accommodate = np.array(hotel_accommodate)

        # user proterty
        self.user_dist = np.array(user_dist) * review_times
        self.user_exp_mean = np.array(user_exp_mean)
        self.user_exp_std = np.array(user_exp_std)

        #self.chosen_hotel = np.random.choice(self.hotel_ids, size=self.n_hotel)
        self.round_cnt = 0
        self.done = False
        self.accom_in = np.array([0 for i in range(self.n_hotel)])
        self.hotel_idx = [i for i in range(self.n_hotel)]
        self.hotel_profit = [0.0 for i in range(self.n_hotel)]
        self.daynum = 365
        self.day_encode = np.identity(self.daynum)


    def reset(self):
        self.round_cnt = 0
        self.done = False
        self.hotel_price = copy.deepcopy(self.init_price)
        self.fit_hotel_price = copy.deepcopy(self.init_price)
        self.accom_in = np.array([0 for i in range(self.n_hotel)])
        self.fit_accom_in = np.array([0 for i in range(self.n_hotel)])

        self.init_state = np.append(self.hotel_price.reshape(self.n_hotel,1),self.day_encode[self.round_cnt]) #np.array([0.0 for i in range(self.n_hotel)], dtype=np.float).reshape(self.n_hotel,1)

        return self.init_state, self.done#self.hotel_price

    def softmax(self,x):
        x -= np.max(x, axis=0, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
        x = np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
        return x


    def step(self,action,dyn_hotels, step):
        # action : price adjustments of chosen hotel

        user_exp = np.random.normal(self.user_exp_mean,
                                    self.user_exp_std, size=self.user_dist[self.round_cnt])
        user_exp[user_exp < self.init_price.min()] = self.init_price.min()
        hotel_score = self.hotel_score / 100.0
        self.accom_in = np.array([0.0 for i in range(self.n_hotel)])
        self.fit_accom_in = np.array([0.0 for i in range(self.n_hotel)])
        self.hotel_price = copy.deepcopy(self.init_price)
        self.fit_hotel_price = copy.deepcopy(self.init_price)
        prob_dyn_price = copy.deepcopy(self.hotel_price)
        prob_fit_price = copy.deepcopy(self.hotel_price)
        whole_action = np.array([0.0 for i in range(self.n_hotel)])
        whole_action[dyn_hotels] = action
        dynamic_price = ((whole_action + 1) * self.hotel_price).reshape(self.n_hotel)

        for user_id in range(self.user_dist[self.round_cnt]):


            prob_dyn_price[(dynamic_price < user_exp[user_id])] = (dynamic_price - user_exp[user_id])[dynamic_price < user_exp[user_id]] / (0.2 * user_exp[user_id])
            prob_dyn_price[(dynamic_price >= user_exp[user_id])] = (dynamic_price - user_exp[user_id])[dynamic_price >= user_exp[user_id]] / (0.1 * user_exp[user_id])
            prob = self.softmax((1-np.abs(prob_dyn_price) + hotel_score).reshape(self.n_hotel)) # user action pattern

            prob_fit_price[(self.fit_hotel_price < user_exp[user_id])] = (self.fit_hotel_price - user_exp[user_id])[self.fit_hotel_price < user_exp[user_id]] / (0.2 * user_exp[user_id])
            prob_fit_price[(self.fit_hotel_price >= user_exp[user_id])] = (self.fit_hotel_price - user_exp[user_id])[self.fit_hotel_price >= user_exp[user_id]] / (0.1 * user_exp[user_id])
            fit_prob = self.softmax((1-np.abs(prob_fit_price) + hotel_score))

            hotel = np.random.choice(self.hotel_idx, 1, p=prob)
            fit_hotel = np.random.choice(self.hotel_idx, 1, p=fit_prob)
            '''
            #if (dynamic_price[hotel] >= 0.5*self.fit_hotel_price[hotel]) and (dynamic_price[hotel] <= 2*self.fit_hotel_price[hotel]):
            self.accom_in[hotel] += 1
            #if (self.fit_hotel_price[fit_hotel] >= 0.5*self.fit_hotel_price[hotel]) and (self.fit_hotel_price[fit_hotel] <= 2*self.fit_hotel_price[hotel]):
            self.fit_accom_in[fit_hotel] += 1

            if self.accom_in[hotel] == self.hotel_accommodate[hotel]:
                dynamic_price[hotel] = 99999
            if self.fit_accom_in[fit_hotel] == self.hotel_accommodate[fit_hotel]:
                self.fit_hotel_price[fit_hotel] = 99999
            '''
            self.accom_in += prob  # [hotel] += 1
            # if (self.fit_hotel_price[fit_hotel] >= 0.5*self.fit_hotel_price[hotel]) and (self.fit_hotel_price[fit_hotel] <= 2*self.fit_hotel_price[hotel]):
            # if (dynamic_price[hotel] <= 2 * self.fit_hotel_price[hotel]):
            self.fit_accom_in += fit_prob

            for i in range(len(prob)):
                if self.accom_in[i] >= self.hotel_accommodate[i]:
                    dynamic_price[i] = 99999
                if self.fit_accom_in[i] >= self.hotel_accommodate[i]:
                    self.fit_hotel_price[i] = 99999
            if self.accom_in.sum() >= self.hotel_accommodate.sum():
                break
        dynamic_profit = (self.accom_in*(whole_action.reshape(self.n_hotel)+1)*self.init_price)[dyn_hotels]
        fit_profit = (self.fit_accom_in*self.init_price)[dyn_hotels]
        reward = ((dynamic_profit-fit_profit)/((self.init_price[dyn_hotels])))#self.fit_accom_in.mean()*

        self.hotel_price = ((whole_action + 1) * self.hotel_price).reshape(self.n_hotel)
        self_state = np.array([self.accom_in[dyn_hotels]/self.hotel_accommodate[dyn_hotels],
                               dynamic_profit/(self.hotel_accommodate[dyn_hotels]*self.init_price[dyn_hotels])])
        next_state = self.hotel_price.reshape(self.n_hotel, 1)#(self.accom_in).reshape(self.n_hotel, 1) #/self.hotel_accommodate
        next_state = (next_state-next_state.min())/(next_state.max()-next_state.min())
        next_state = np.append(next_state, self.day_encode[self.round_cnt])

        self.round_cnt += 1
        if self.round_cnt == self.num_steps:
            self.done = True

        return next_state.astype(float),self_state.astype(float),reward,(dynamic_profit-fit_profit),fit_profit,self.done

