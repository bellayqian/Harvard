from itertools import product
import numpy as np
from numpy.random import default_rng
from pandas import DataFrame
import os

np.random.seed(42)
storage_path='./Data/'

def h(t): # customer’s price sensitivity
    return 2 * ((t - 5) ** 4 / 600 + np.exp(-4 * (t - 5) ** 2) + t / 10 - 2)

def f(p, t, s):
    return 100 + (10 + p) * s * h(t) - 2 * p

def generate_Demand_train(data_size=10000, rho=0.5, alpha=1, beta=0, rng=default_rng(2021)):
    
    emotion = rng.choice(list(range(1, 8)), (data_size,1))
    time = rng.uniform(1, 12, (data_size,1)) # time of year
    cost = rng.normal(0, 1.0, (data_size,1)) 
    noise_price = rng.normal(0, 1.0, (data_size,1))
    noise_demand = rho * noise_price + rng.normal(0, np.sqrt(1 - rho ** 2), (data_size,1))
    price_sensitivity = h(time)
    price = 25 + (alpha * cost + 3) * h(time) + beta * cost + noise_price
    structural = f(price, time, emotion).astype(float)
    outcome = (structural + noise_demand).astype(float)
    
    mu0 = f(price-price, time, emotion).astype(float)
    mut = structural
    
    numpys = [noise_price,noise_demand, cost, time, emotion, price, mu0, mut, structural, outcome, price_sensitivity]
    
    train_data = DataFrame(np.concatenate(numpys, axis=1),
                          columns=['u1','u2','v1','x1','x2','t1','m0','mt','g1','y1','x3'])
    
    return train_data
'''
def generate_Demand_test(rho=0.5, alpha=1, beta=0, rng = default_rng(2021)):
    
    noise_price = rng.normal(0, 1.0, (2800,1))
    noise_demand = rho * noise_price + rng.normal(0, np.sqrt(1 - rho ** 2), (2800,1))
    
    cost = np.linspace(-1.0, 1.0, 20)
    time = np.linspace(0.0, 10, 20)
    emotion = np.array([1, 2, 3, 4, 5, 6, 7])
    
    data = []
    price_z = []
    for c, t, s in product(cost, time, emotion):
        data.append([c, t, s])
        price_z.append(25 + (alpha * c + 3) * h(t) + beta * c)
    features = np.array(data)
    price_z = np.array(price_z)[:, np.newaxis]
    price = price_z + noise_price
    
    structural = f(price, features[:,1:2], features[:,2:3]).astype(float)
    outcome = (structural + noise_demand).astype(float)
    price_sensitivity = h(time)

    mu0 = f(price-price, features[:,1:2], features[:,2:3]).astype(float)
    mut = structural
    
    numpys = [noise_price, noise_demand, cost, features[:,1:2], features[:,2:3], price, mu0, mut, structural, outcome, price_sensitivity]
    
    test_data = DataFrame(np.concatenate(numpys, axis=1),
                          columns=['u1','u2','v1','x1','x2','t1','m0','mt','g1','y1','x3'])
    
    return test_data
'''

def generate_Demand_test(rho=0.5, alpha=1, beta=0, rng = default_rng(2021)):
    
    noise_price = rng.normal(0, 1.0, (2800,1))
    noise_demand = rho * noise_price + rng.normal(0, np.sqrt(1 - rho ** 2), (2800,1))
    
    cost = np.linspace(-1.0, 1.0, 20)
    time = np.linspace(0.0, 10, 20)
    emotion = np.array([1, 2, 3, 4, 5, 6, 7])
    
    data = []
    price_z = []
    for c, t, s in product(cost, time, emotion):
        data.append([c, t, s])
        price_z.append(25 + (alpha * c + 3) * h(t) + beta * c)
    features = np.array(data)
    price_z = np.array(price_z)[:, np.newaxis]
    price = price_z + noise_price
    price_sensitivity = h(time)
    
    structural = f(price, features[:,1:2], features[:,2:3]).astype(float)
    outcome = (structural + noise_demand).astype(float)
    
    mu0 = f(price-price, features[:,1:2], features[:,2:3]).astype(float)
    mut = structural
    
    sett = [noise_price, noise_demand, features, price, mu0, mut, structural, outcome, 
              price_sensitivity]
    
    test_data = DataFrame(np.concatenate(sett, axis=1),
                          columns=['u1','u2','v1','x1','x2','t1','m0','mt','g1','y1','x3'])
    
    return test_data

class Demand_Generator(object):
    def __init__(self, num=10000, num_reps=10, seed=2021, storage_path='./Data/'):
        self.num = num
        self.num_reps = num_reps
        self.seed = seed
        self.storage_path = storage_path
        
        self.set_path()
        self.run()
        
    def set_path(self):
        self.data_path = self.storage_path + 'data/Demand/'
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
    def run(self):
        
        print('Next, run dataGenerator: ')
        
        for exp in range(self.num_reps):
            seed = exp * 527 + self.seed
            print(f'Generate Demand datasets - {exp}/{self.num_reps}. ')
            train_df = generate_Demand_train(data_size=10000, rho=0.5, rng=default_rng(seed))
            valid_df = generate_Demand_train(data_size=10000, rho=0.5, rng=default_rng(seed+1111))
            test_df = generate_Demand_test(rho=0.5, rng=default_rng(seed+2222))
            
            data_path = self.data_path + '/{}/'.format(exp)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            
            train_df.to_csv(data_path + '/train.csv', index=False)
            valid_df.to_csv(data_path + '/val.csv', index=False)
            test_df.to_csv(data_path + '/test.csv', index=False)
        
        print('-'*30)
        
Demand = Demand_Generator(storage_path=storage_path)