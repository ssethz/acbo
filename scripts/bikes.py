"""
Environments for the bikes experiments.
I would first recommend looking at functions.py since the class structure is similar 
and the environments are much simpler. 
"""
try:
    from scripts.functions import FNEnv
except:
    from functions import FNEnv
from mcbo.utils.dag import DAG, FNActionInput
from mcbo.utils import functions_utils
import torch
import math
import numpy as np
from copy import deepcopy
import geopy.distance
from scipy.spatial import distance
import pandas as pd
import pickle

class Rentals_Simulator:
    """ Class used to simulate rentals in the city from historic data. """

    def __init__(self, trips_data, centroids, depth=2, walk_dist_max = 1):
        self.trips_data = trips_data  # history of trips data (used to simulate rentals)
        self.centroids = centroids  # centroids for regions in the city where bikes/scooters are positioned
        self.R = len(centroids)  # number of regions in the city
        self.chunks = depth +1 #number of time chunks to break the day into. You add 1 to the depth
        self.walk_dist_max = walk_dist_max

    def simulate_rentals(self, X, daynum, month):
        """ 
        Simulate daily rentals when X[i] bikes are positioned in each region i at the beginning of the day on daynum of month.
        Returns a list of the starting coordinates of the trips that were met and a list of the starting coordinates of the trips that were unmet.
        """
        query = 'Day == ' + str(daynum) + '& Month == ' + str(month)
        daily_trips = self.trips_data.query(query)
        X_copy = np.array(X)
        R = len(self.centroids)
        tot_trips = np.zeros(R)
        trips_starting_coords = []
        trips_unmet_coords = []
        trips_met_times = []
        unmet_demand = np.zeros(R)
        L = len(daily_trips)

        TRIP_DURATION = 0.5 #in hours: the time a bike is removed from the system for while a trip is happening

        taken_bikes = [] 
        # a list of tuples. Each tuple comtains time the bike will be returned and the location it will be returned to

        current_chunk = 0 
        # divide the 24 hours up into chunks and put these in a list
        chunk_size = 24 / (self.chunks)
        threshold = chunk_size

        trips_per_chunk = np.zeros(self.chunks) # trips in each chunk of time
        trips_per_chunk_per_centroid = np.zeros((R, self.chunks)) #trips in each centroid at each chunk of time
        unmet_trips_per_chunk = np.zeros(self.chunks) # trips that were not met in each chunk of time
        x_new_chunks = np.zeros((self.chunks-1, self.R)) #x_new_chunks is the number of bikes at each centroid at the end of each chunk of time
        total_walking_distance = 0

        for d in range(L):
            trip = d
            start_loc = np.array([daily_trips.iloc[trip].StartLatitude, daily_trips.iloc[trip].StartLongitude])
            start_time = daily_trips.iloc[trip].StartTime

            #this is a str in 24 hour format. convert to a float in hours
            start_time = float(start_time[0:2]) + float(start_time[3:5])/60

            #check if any bikes that were in transit have completed there trip, and add them back to the system
            for bike in taken_bikes:
                if bike[0] <= start_time:
                    X_copy[bike[1]] += 1
                    taken_bikes.remove(bike)

            if start_time >= threshold and start_time != 24.0:
                # If a certain amount of time has passed we move on to the next chunk of time
                x_new_chunks[current_chunk, :] = X_copy
                threshold += chunk_size
                current_chunk += 1 
                
            distances = distance.cdist(start_loc.reshape(-1, 2), self.centroids, metric='euclidean')
            idx_argsort = np.argsort(distances)[0]
            trip_met = 0
            # We go through the centroids in order of closest to farthest and see if there is an available bike at one of the centroids to make the trip
            for idx_start_centroid in idx_argsort:
                walk_dist_max = self.walk_dist_max
                if geopy.distance.distance(start_loc, self.centroids[idx_start_centroid, :]).km > walk_dist_max or trip_met == 1:
                    break
                if X_copy[idx_start_centroid] > 0:
                    # If the trip can be met, we update all of the relevent lists tracking trips and where bikes are
                    X_copy[idx_start_centroid] -= 1
                    tot_trips[idx_start_centroid] += 1  # daily_trips.iloc[trip].TripDistance
                    end_loc = np.array([daily_trips.iloc[trip].EndLatitude, daily_trips.iloc[trip].EndLongitude])
                    distances = distance.cdist(end_loc.reshape(-1, 2), self.centroids, metric='euclidean')
                    idx_end_centroid = np.argmin(distances)
                    # idx_end_centroid = np.argmin(distance.cdist(end_loc.reshape(-1,2), centroids, metric='euclidean'))
                    total_walking_distance += geopy.distance.distance(start_loc, self.centroids[idx_start_centroid, :]).km
                    #X_copy[idx_end_centroid] += 1
                    taken_bikes.append((start_time + TRIP_DURATION, idx_end_centroid))
                    trip_met = 1
                    trips_starting_coords.append(start_loc)
                    trips_met_times.append(current_chunk)
                    trips_per_chunk[current_chunk] += 1
                    trips_per_chunk_per_centroid[idx_start_centroid, current_chunk] += 1
            if trip_met == 0:
                # If the trip is unmet record it in the relevent lists
                unmet_demand[idx_argsort[0]] += 1
                unmet_trips_per_chunk[current_chunk] += 1
                trips_unmet_coords.append(start_loc)
        #Compute the final bike locations at the end of the day (unused)
        if self.chunks > 1:
            x_new_chunks[-1, :] = X_copy
        return trips_starting_coords, trips_unmet_coords, x_new_chunks

class Bikes(FNEnv):
    """
    A basic bikes class that is used as a base for the BikesSparse class used in our experiments. It is significantly more general than BikesSparse. 
    Does not contain an evaluate function because it is only meant to be subclassed.
    """
    def __init__(self, depth = 2, centroids = 116, N=5, chunk_demand = True, alpha = 0.01, split_reward_by_centroid = True, walk_distance_max = 1.0):
        # N is the number of trucks
        self.N = N
        self.depth = depth # depth is the depth of the graph. If depth is 1, each node is the total trips for a given region in the day. If depth is two, we have seperate nodes for before and after noon, etc.
        self.centroids = centroids # the number of bike depots
        self.split_reward_by_centroid = split_reward_by_centroid # if true, we model the number of trips from each centroid individual, instead of the total number of trips

        self.alpha = alpha # threshold for determining graph edges from the trip data. If the number of trips from region i to region j is greater than alpha, then there is an edge between trips from region i at time t to j at time t+1

        # load centroids_trips_matrix5_40 from pckl. This is the number of trips betwen each pair of regions (normalized)
        self.centroid_trips_matrix = pickle.load(open("scripts/bikes_data/centroid_trips_matrix5_40.pckl", "rb" ))

        self.parent_nodes = self.get_parent_nodes()
        self.bikes_per_truck = 8 # the number of bikes dropped-off by each truck
        self.n_bikes = self.N * self.bikes_per_truck
        self.chunk_demand = chunk_demand # whether to use a finer granularity when reporting demand information (demand per centroid) or just the total demand
        
        dag = DAG(self.parent_nodes)
        
        #self.input_dim_a = 1
        active_input_indices, full_input_indices = self.adapt_active_input_indices()
        self.active_input_indices = active_input_indices

        self.input_dim_a = N

        action_input = FNActionInput(active_input_indices)
        full_input = FNActionInput(full_input_indices)

        discrete = 2 # doesn't matter because we have discrete player input and continuous adversary input, so we end up overriding this

        noise_scales = 0.0
        
        super(Bikes, self).__init__(dag, action_input, discrete=discrete, input_dim_a = self.input_dim_a, full_input=full_input)

        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )  

        trips_data = pd.read_csv("scripts/bikes_data/dockless-vehicles-3_full.csv", usecols= lambda x: x not in ["TripID", "StartDate", "EndDate", "EndTime"])
        weather_data = pd.read_csv("scripts/bikes_data/weather_data.csv", usecols=["Year", "Month", "Day", "DayOfWeek", "Temp_Avg", "Precip", "Holiday"])
    
        hour_max = 24

        period = 'Month > 0 & Month < 13 & Year == 19 & DayOfWeek >=0 and DayOfWeek <=8'
        area =  'StartLatitude < 38.35 & StartLatitude > 38.15 & StartLongitude < -85.55 & StartLongitude > -85.9 ' \
                '& EndLatitude < 38.35 & EndLatitude > 38.15 & EndLongitude < -85.55 & EndLongitude > -85.9 '
        query = 'TripDuration < 60 & TripDuration > 0 & HourNum <= '+ str(hour_max) + '' \
                '&' + area  + ' & ' + period
        trips_data = trips_data.query(query)
        
        weather_data = weather_data.query(period+ "& Holiday == 0")

        # take out all weather data that corresponds to a weekend. 1.0 is a sunday and 7.0 is a saturday in the dataset
        weekdays = [2,3,4,5,6]
        weather_data = weather_data.query('DayOfWeek in @weekdays')
        # don't need to filter the trips data because we use the weather data to look-up what trips data to use

        self.weather_data = weather_data
        # get the coordinates of the depots
        _, _, _, _, _, _, centroid_coords = pickle.load(open( "scripts/bikes_data/training_data_" + '5' + '_' + '40' + ".pckl", "rb" ))
        R = len(centroid_coords)
        
        self.centroid_coords = centroid_coords

        self.sim = Rentals_Simulator(trips_data, centroid_coords, depth=depth, walk_dist_max = walk_distance_max)

        # size of the inputs not controlled by the truck (weather, demand)
        self.z_shape = 3
        if self.chunk_demand:
            self.z_shape = self.z_shape + self.depth +1

        self.z_max = None

        self.z_max = self.get_z_max()

    def get_env_profile(self):
        """
        Outputs a dictionary containing all details of the environment needed for 
        BO experiments. We override this from the base class to append some bikes-specific properties. 
        """
        # bikes_quantities are properties specific to the bikes to be appended to the env_profile dict. include the centroids
        bikes_quantities = {'depth': self.depth, 'centroids': self.centroids, 'n_bikes': self.n_bikes, 'bikes_per_truck': self.bikes_per_truck, 'z_shape': self.z_shape, "trucks": self.N, "walk_distance_max": self.sim.walk_dist_max, "z_max": self.z_max, "centroid_coords": self.sim.centroids}

        return {**self.get_base_profile(), **self.get_causal_quantities(), **bikes_quantities}

    def get_parent_nodes(self):
        """
        Computes a list of lists of the parent nodes for every node. Uses get_parets_row as a helper function to construct this list
        for nodes at each depth. Implemented in sublasses. 
        """
        raise NotImplementedError
    
    def adapt_active_input_indices(self):
        """
        Computes for every node which input actitions are parents.
        full_input_indices includes actions not by the trucks. 
        Implemented in subclasses. 
        """
        raise NotImplementedError 
    
    def get_z_max(self):

        # just run get_z_t for every t and take the max
        z_max = np.zeros(self.z_shape)
        for t in range(len(self.weather_data)):
            z_t = self.get_z_t(t)
            z_max = np.maximum(z_max, z_t)
        return z_max
    
    def get_demand_max(self):
        """
        Returns just the max demand for a day. Used for normalizing rewards.
        """
        return self.z_max[2]

    def get_z_t(self, t):
        """
        Given a time t generates the z actions (adversary actions)
        """
        # if t is greater than weather data length throw an error
        assert t < len(self.weather_data), "t is greater than weather data length"

        day_t = np.mod(t, len(self.weather_data))
        daynum = self.weather_data.iloc[day_t].Day
        month = self.weather_data.iloc[day_t].Month
        dayofweek = self.weather_data.iloc[day_t].DayOfWeek

        query = 'Day == ' + str(daynum) + '& Month == ' + str(month)
        weather = self.weather_data.query(query)
        demand = len(self.sim.trips_data.query(query))
        # I also want to get the demand at the first, second third etc self.depth chunk of the day
        # so I will divide the day into self.depth chunks and get the demand in each chunk

        start_times = self.sim.trips_data.query(query).StartTime
        # convert this series from string time format to float in number of hours
        start_times = start_times.apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60)
        
        demand_i = []
        # count elements in start_times less than 24/self.depth, 2*24/self.depth, 3*24/self.depth etc
        for i in range(self.depth+1):
            if i == 0:
                demand_i.append(len(start_times[start_times <= 24/(self.depth+1)]))
            else:
                demand_i.append(len(start_times[(start_times > i*24/(self.depth+1)) & (start_times <= (i+1)*24/(self.depth+1))]))
        
        if self.chunk_demand:
            z_t = np.concatenate([weather.Temp_Avg, weather.Precip,[demand], demand_i])
        else:
            z_t = np.concatenate([weather.Temp_Avg, weather.Precip, [demand]])

        assert z_t.shape[0] == self.z_shape, "z_t shape is not correct"

        if self.z_max is None:
            return z_t
        
        # asseert shape the same
        assert z_t.shape == self.z_max.shape, "z_t shape is not correct"
        return z_t / self.z_max

    def evaluate(self, X, X_a, t):
        """
        For a given t, bike assignment X, and adversary action X_a, compute the rewards and intermediate values in the graph. 
        """
        raise NotImplementedError

class BikesSparse(Bikes):
    """
    A sparse version of the bikes environment where we consider only depth 1 and the number of trips in each region only depends on bikes 
    placed at depots within that region. This information is embedded in the parent nodes.
    """
    def __init__(self, depth=1, centroids=116, N=5, chunk_demand=False, alpha=0.01, split_reward_by_centroid=True, walk_distance_max = 1.0, norm_reward=False):
        #Depth basically doesn't matter since we only use depth = 1, but the parent class can have depth > 1.
        self.norm_reward = norm_reward
        self.num_clusters = 15
        self.cluster_labels, self.cluster_centers = pickle.load(open("scripts/bikes_data/clustered_centroids_" + str(centroids) + '_' + str(self.num_clusters) + ".pckl", "rb" ))
        
        super().__init__(depth, centroids, N, chunk_demand, alpha, split_reward_by_centroid, walk_distance_max)
        
    def get_parent_nodes(self):
        """
        A list for each node of the indices of other nodes that are parents of that node. 
        For sparse graph all nodes except the reward have no parents besides the inputs. 
        """
        x = [[] for i in range(self.num_clusters)]
        x.append([i for i in range(self.num_clusters)])
        return x

    def adapt_active_input_indices(self):
        """
        active_input_indices is a list for each node of the input indices that are parents of that node.
        full_input_indices adds any non-player inputs to these lists. 
        """
        active_input_indices = []
        full_input_indices = []

        for i in range(self.num_clusters):
            # whether an input causes a node is determined by whether they the centroid corresponding to that node is in the same region/cluster
            active_input_indices.append([j for j in range(self.centroids) if self.cluster_labels[j] == i])
            full_input_indices.append([j for j in range(self.centroids) if self.cluster_labels[j] == i])
            # add to full_input the adversary values (weather and demand)
            full_input_indices[i] = full_input_indices[i] + [j for j in range(self.centroids, self.centroids+3)] #weather data

        active_input_indices.append([])
        full_input_indices.append([j for j in range(self.centroids, self.centroids+3)])

        return active_input_indices, full_input_indices
    
    def evaluate(self, X, X_a, t):
        return self.evaluate_with_trip_coords(X, X_a, t)[0]
    
    def evaluate_with_trip_coords(self, X, X_a, t):
        # if t is greater than weather data length throw an error
        assert t < len(self.weather_data), "t is greater than weather data length"

        day_t = np.mod(t, len(self.weather_data))
        daynum = self.weather_data.iloc[day_t].Day
        month = self.weather_data.iloc[day_t].Month
        dayofweek = self.weather_data.iloc[day_t].DayOfWeek

        query = 'Day == ' + str(daynum) + '& Month == ' + str(month)
        weather = self.weather_data.query(query)
        demand = len(self.sim.trips_data.query(query))

        z_t = self.get_z_t(t)

        # Transform X to an np.array with the number of bikes at each centroid
        X_sim_input = np.zeros(len(self.sim.centroids))
        for i in X:
            X_sim_input[i] += self.bikes_per_truck

        trips_starting_coords, trips_unmet_coords, x_new_chunks = self.sim.simulate_rentals(X_sim_input, daynum, month)
        
        # Go through trips_starting_coords, find the closest cluster to each (euclidean distance) and count them towards that cluster

        trips_per_cluster = np.zeros(self.num_clusters)
        for trip in trips_starting_coords:
            # find the closest cluster
            closest_cluster = 0
            closest_centroid = 0
            closest_d = 1000000
            for i in range(self.centroids):
                d = np.linalg.norm(trip - self.centroid_coords[i])
                if d < closest_d:
                    closest_d = d
                    closest_centroid = i
            closest_cluster = self.cluster_labels[closest_centroid]
            trips_per_cluster[closest_cluster] += 1
        
        trips_per_cluster = trips_per_cluster / self.get_demand_max()
        
        reward = np.sum(trips_per_cluster)
        output = np.concatenate([trips_per_cluster.flatten(), reward.flatten()]), trips_starting_coords, trips_unmet_coords, x_new_chunks

        return output
    
    def get_max_per_node(self):
        """
        Calls evaluate with a packed X (bikes everywhere) for every t then takes a max to get the max per node
        """
        max_per_node = np.zeros(self.num_clusters+1)
        for t in range(len(self.weather_data)):
            round_t_eval = self.evaluate(5 * [i for i in range(self.centroids)], 1, t)

            max_per_node = np.maximum(max_per_node, round_t_eval)
        return max_per_node

if __name__ == "__main__":
    """
    Prints the maximum of each node in a bikes environment. 
    """
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=2)
    args = parser.parse_args()
    depth = args.depth

    env = BikesSparse(depth = depth, alpha=-0.1)
    max_per_node = env.get_max_per_node()
    print(max_per_node)
    