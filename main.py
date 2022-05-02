import argparse
import pickle
import model
import time


parser = argparse.ArgumentParser()

#Global property
parser.add_argument('--region', type=int, default=4)
parser.add_argument('--num_dynamic_agent', type=int, default=10)

#Env
parser.add_argument('--max_step', type=int, default=365)
parser.add_argument('--review_times', type=int, default=15)

#DQN
parser.add_argument('--tau', type=float, default=0.01)
parser.add_argument('--p_lr', type=float, default=1e-3) #0.0005 for whole; 0.0002 for partial
parser.add_argument('--p_lr2', type=float, default=1e-7)
parser.add_argument('--v_lr', type=float, default=1e-3) # 0.001 for whole; 0.0005 for partial
parser.add_argument('--v_lr2', type=float, default=1e-7)
parser.add_argument('--memory_size', type=int, default=30000)
parser.add_argument('--eps_start', type=float, default=0.9)
parser.add_argument('--eps_end', type=float, default=0.1)
parser.add_argument('--eps_decay', type=float, default=2e-3)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--double_q', type=bool, default=True)
parser.add_argument('--layer', type=int, default=2)
parser.add_argument('--device', type=int, default=0)

#train
parser.add_argument('--batch_size',type=int, default=500)
parser.add_argument('--train_times', type=int, default=3, help='total epoch')

parser.add_argument('--sample_times', type=int, default=2, help='sample epoch times')
parser.add_argument('--update', type=int, default=1000, help='learning epoch times')


args = parser.parse_args()

filename = "dataFile.txt"

fr = open(filename,'rb')

datasets = pickle.load(fr)
hotels = datasets[0]
n_hotels = [len(h) for h in hotels]

#for i in range(2,10):
#    print("Region : " + str(i))
#    args.region = i
    #args.num_dynamic_agent = n_hotels[i]

dyn_model = model.DynamicPricing(args, datasets)

dyn_model.train()



