import heapq
import random
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.stats import t

"""
Class to characterize an event.

Attributes:
    time: at which time the event occurs
    event_type: 'arrival' or 'departure' (also 'end' to manage the end of a simulation)
"""
class Event:
    def __init__(self, time, event_type):
        self.time = time
        self.event_type = event_type

    def __lt__(self, other):
        return self.time < other.time

"""
Class to perform the simulation.

Attributes:
    arrival_rate: λ
    service_rate: μ
    simulation_time: for how long we make the simulation last
    
    event_queue: store the events scheduled (it's a priority queue where priority is based on time and we use a binary heap to manage it)
    current_time: current time of the simulation
    server_busy: True or False to determine is the server is busy or not
    queue_length: we save the number of packets in queue
    packets_in_system: packets in the system at a given time
"""
class MM1QueueSimulator:
    
    def __init__(self, arrival_rate, service_rate, simulation_time):
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.simulation_time = simulation_time
        self.event_queue = []
        self.current_time = 0
        self.server_busy = False
        self.queue_length = 0
        self.packets_in_system = []

        self.waiting_times = []
        self.empirical_service_times = []
        self.empirical_arrival_times = []

    """
    Add an event in queue.
    """
    def schedule_event(self, event):
        heapq.heappush(self.event_queue, event)

    """
    Method to handle the simulaton
    """
    def simulate(self):
        #schedule the first arrival and simulation end
        self.schedule_event(Event(self.generate_time(self.arrival_rate), "arrival"))
        self.schedule_event(Event(self.simulation_time, "end"))

       # print(f"Simulation started at time {self.current_time}")

        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if event.event_type == "arrival":
                #print(f"Arrival at time {self.current_time} [in system {self.queue_length + int(self.server_busy)}]")
                self.handle_arrival()
            elif event.event_type == "departure":
                #print(f"Departure at time {self.current_time} [in system {self.queue_length + int(self.server_busy)}]")
                self.handle_departure()
            elif event.event_type == "end":
                #print(f"Simulation ended at time {self.current_time} [in system {self.queue_length + int(self.server_busy)}]. I still could need to handle some last departure events")
                break

        #process last departures in the queue
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if event.event_type == "arrival":
                print(f"SHOULD NOT HAPPEN")
                self.handle_arrival()
            elif event.event_type == "departure":
               # print(f"Departure at time {self.current_time} [in system {self.queue_length + int(self.server_busy)}]")
                self.handle_departure()
            elif event.event_type == "end":
                print(f"SHOULD NOT HAPPEN")
        
        self.packets_in_system.append((self.current_time, self.queue_length + int(self.server_busy)))




    def handle_arrival(self):
        #record the number of packets in the system: we store the packest in the system at the current time
        self.packets_in_system.append((self.current_time, self.queue_length + int(self.server_busy)))
        self.empirical_arrival_times.append(self.current_time)

        if self.server_busy:
            self.queue_length += 1
        else:
            self.server_busy = True
            self.empirical_service_times.append(self.current_time)
            self.schedule_event(Event(self.current_time + self.generate_time(self.service_rate), "departure"))

        #schedule the next arrival - only if it can arrive until the end of the simualtion
        next_arrival = self.current_time + self.generate_time(self.arrival_rate)
        if next_arrival < self.simulation_time:
            self.schedule_event(Event(next_arrival, "arrival"))

    def handle_departure(self):
        #record the number of packets in the system: we store the packest in the system at the current time
        self.packets_in_system.append((self.current_time, self.queue_length + int(self.server_busy)))

        if self.queue_length > 0:
            self.queue_length -= 1
            self.empirical_service_times.append(self.current_time)
            self.schedule_event(Event(self.current_time + self.generate_time(self.service_rate), "departure"))
        else:
            self.server_busy = False

    #draw from an exponential
    def generate_time(self, rate):
        return random.expovariate(rate)

    def plot_results(self):
        times, packets = zip(*self.packets_in_system)
        plt.plot(times, packets, drawstyle='steps-post')
        plt.xlabel("Time")
        plt.ylabel("Number of Packets in System")
        plt.title("Number of Packets in System Over Time")
        plt.show()
    

    """
    Method to compute the average empirically from the data
    """
    def compute_average(self):
        
        total_time = 0
        weighted_sum = 0

        for i in range(len(self.packets_in_system) - 1):
            time_current, packets_current = self.packets_in_system[i]
            time_next, _ = self.packets_in_system[i + 1]

            #time interval
            delta_time = time_next - time_current

            #add to weighted sum
            weighted_sum += packets_current * delta_time

            total_time += delta_time    
        
        average_packets = weighted_sum / total_time
        return average_packets

    """
    Method to compute the average empirically from the data, with a warmup period
    """
    def compute_average_with_warmup(self, warmup_time=0.1):
        
        last_event_time, _ = self.packets_in_system[len(self.packets_in_system) - 1]
        warmup_limit = last_event_time * warmup_time
        filtered_data = [(t, p) for t, p in self.packets_in_system if t >= warmup_limit]

        total_time = 0
        weighted_sum = 0

        for i in range(len(filtered_data) - 1):
            time_current, packets_current = filtered_data[i]
            time_next, _ = filtered_data[i + 1]

            #time interval
            delta_time = time_next - time_current

            #add to weighted sum
            weighted_sum += packets_current * delta_time

            total_time += delta_time    
        
        average_packets = weighted_sum / total_time
        return average_packets

    def compute_average_waiting_time(self):
        for i in range(len(self.empirical_service_times)):
            self.waiting_times.append(self.empirical_service_times[i] - self.empirical_arrival_times[i])
    
    def plot_waiting_times(self):
        self.compute_average_waiting_time()

        points = np.linspace(0, self.simulation_time + 2, len(self.waiting_times))

        plt.hist(self.waiting_times, color="red", density=True)
        #plt.plot(points, list(map(lambda x : 1/(self.service_rate - self.arrival_rate), points)), color="black")
        plt.xlabel("Time")
        plt.ylabel("Waiting time in System")
        plt.title("Waiting time in System Over Time")
        plt.show()



def confidence_interval(data, confidence_level=0.95):
    data_mean = np.mean(data)
    replications = len(data)
    variance_estimator = sum([ ((data[i] - data_mean)**2) for i in range(replications)])
    variance_estimator = variance_estimator / (replications - 1)

    degrees_of_freedom = replications - 1
    alpha = (1 + confidence_level) / 2
    t_quantile = t.ppf(alpha, degrees_of_freedom)
    lower_bound = data_mean - t_quantile * np.sqrt(variance_estimator / replications)
    upper_bound = data_mean + t_quantile * np.sqrt(variance_estimator / replications)
    return lower_bound, upper_bound


if __name__ == "__main__":
    #parameters
    arrival_rate = 1.0  #λ
    service_rate = 2.0  #μ
    simulation_time = 1000.0  #seconds
    replications = 500

    parser = argparse.ArgumentParser(description="M/M/1 Queue Simulation")
    parser.add_argument("--arrival_rate", type=float, default=arrival_rate, help=f"Arrival rate (λ, default {arrival_rate})")
    parser.add_argument("--service_rate", type=float, default=service_rate, help=f"Service rate (μ, default {service_rate})")
    parser.add_argument("--simulation_time", type=float, default=simulation_time, help=f"Time of the simulation (default {simulation_time})")
    parser.add_argument("--replications", type=int, default=replications, help=f"Number of independent replications we do (default {replications})")
    args = parser.parse_args()

    arrival_rate = args.arrival_rate
    service_rate = args.service_rate
    if(service_rate <= arrival_rate):
        print("Cannot continue!  Must have μ > λ!")
        exit(0)
    simulation_time = args.simulation_time
    replications = args.replications

    print(f"You are running the simulator with an arrival_rate of {arrival_rate}, a service_rate of {service_rate} and a simulation_time of {simulation_time}, for {replications} independent replications")

    #We do independent replications
    empirical_averages = []
    empirical_averages_warmup = []

    for j in range(replications):
        simulator = MM1QueueSimulator(arrival_rate, service_rate, simulation_time)
        simulator.simulate()
        #simulator.plot_results()
        #simulator.plot_waiting_times()

        #Comparisons
        rho = arrival_rate / service_rate
        theoretical_avg = rho / (1 - rho)
        #print(f"Theoretical average number of packets in system: {theoretical_avg}")
        empirical_avg = simulator.compute_average()
        #print(f"Empirical average number of packets in system: {empirical_avg}")
        empirical_averages.append(empirical_avg)
        empirical_avg_warmup = simulator.compute_average_with_warmup(warmup_time=0.5)
        empirical_averages_warmup.append(empirical_avg_warmup)

    rho = arrival_rate / service_rate
    theoretical_avg = rho / (1 - rho)
    print(f"\nTheoretical average number of packets in system: {theoretical_avg}\n")
    
    print(f"Empirical average number of packets in system: {np.mean(empirical_averages)} (standard deviation {np.std(empirical_averages)})")
    lower_ci, upper_ci = confidence_interval(empirical_averages)
    print(f"Confidence Interval: [{lower_ci}, {upper_ci}]\n")
    
    print(f"[WITH WARMUP] Empirical average number of packets in system: {np.mean(empirical_averages_warmup)} (standard deviation {np.std(empirical_averages_warmup)})")
    lower_ci, upper_ci = confidence_interval(empirical_averages_warmup)
    print(f"Confidence Interval: [{lower_ci}, {upper_ci}]\n")

