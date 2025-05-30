import numpy as np
import matplotlib.pyplot as plt

arrival_rate = 1   #lambda
departure_rate = 4 #mu

#mu > lambda

#both arrivals are POISSON DISTRIBUTION
#inter-arrivals and inter-departures are EXPONENTIALLY DISTRIBUTED

#Estimate:
"""
- average packets in queue
- average packets in server
- average usage of system (sum)
"""

#one server, one queue with FIFO policy

#start of simulation
#end of simulation
#arrival of packet
#departure of packet

#ordered queue/list where
"""
- event linked to the next in time
- e1 at t1, e2 at t2, e3 at t3:
- if there is e4 at t4 s.tt t2 < t4 < t3 i put: e1 e2 e4 e3
"""

def compute_event_queue(arrival_rate, service_rate, N):
    
    #inter-arrival times
    inter_arrival_times = np.random.exponential(1 / arrival_rate, N)
    
    #service times
    service_times = np.random.exponential(1 / service_rate, N)

    #arrival times
    arrival_times = np.cumsum(inter_arrival_times)

    #departure times
    departure_times = np.zeros(N)

    for i in range(N):
        if i == 0:
            departure_times[i] = arrival_times[i] + service_times[i]
        else:
            departure_times[i] = max(arrival_times[i], departure_times[i - 1]) + service_times[i]
            #depending if server is free when event arrives or it's busy, so it must way to i-1 event to go

    arrivals_index = 0
    departure_index = 0
    event_queue = []
    
    while arrivals_index < N and departure_index < N:
        if arrival_times[arrivals_index] < departure_times[departure_index]:
            event_queue.append((float(arrival_times[arrivals_index]), 'a'))
            arrivals_index += 1
        else:
            event_queue.append((float(departure_times[departure_index]), 'd'))
            departure_index += 1

    while arrivals_index < N:
        print("it should not happen")
        arrivals_index += 1

    while departure_index < N:
        event_queue.append((float(departure_times[departure_index]), 'd'))
        departure_index += 1

    return event_queue

events_list = compute_event_queue(arrival_rate, departure_rate, 50)

def simulate_with_list(events_list):
    
    current_time = 0
    index_event = 0

    server_status = 0
    in_queue = 0
    last_event_time = 0

    time_server_full = 0
    area_server = 0
    area_queue = 0

    server_status_history = [0]
    queue_length_history = [0]
    times = [0]
    system_history = [0]
    area_system = 0


    while index_event < len(events_list):
        event = events_list[index_event]
        index_event += 1

        current_time = event[0]
        type_event = event[1]

        
        area_server += (current_time - last_event_time) * server_status
        area_queue += in_queue * (current_time - last_event_time)
        area_system += in_queue * (current_time - last_event_time) + (current_time - last_event_time) * server_status


        if type_event == 'a':
            if server_status == 0:
                server_status = 1
            else:
                in_queue += 1
        else:
            
            server_status = 0

            if in_queue != 0:
                server_status = 1
                in_queue -= 1

        print(f"Area Q(t) {area_queue:.3f}, Area U(t) {area_server:.3f}, Area system {area_system:3f}")
        
        times.append(current_time)
        server_status_history.append(server_status)
        queue_length_history.append(in_queue)
        system_history.append(server_status+in_queue)

        last_event_time = current_time

    plot = False

    if(plot):
        p = arrival_rate/departure_rate
        print(f"Average number of packets in the system {(p*(1-p)):.3f}")
        print(f"Empirical number of packets in the system {(area_system / current_time):.3f}")

        max_y = max(max(system_history), max(queue_length_history), max(server_status_history))

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.plot(times, server_status_history, drawstyle='steps-post')
        plt.xlabel("Time")
        plt.ylabel("Server status")
        plt.yticks(np.arange(0, 3, 1)) 
        plt.title("Server Status Over Time")

        plt.subplot(1, 3, 2)
        plt.plot(times, queue_length_history, drawstyle='steps-post', color='orange')
        plt.xlabel("Time")
        plt.ylabel("Queue length")
        plt.yticks(np.arange(0, max_y+2, 1))
        plt.title("Queue Length Over Time")

        plt.subplot(1, 3, 3)
        plt.plot(times, system_history, drawstyle='steps-post', color='green')
        plt.xlabel("Time")
        plt.ylabel("System packets")
        plt.yticks(np.arange(0, max_y+2, 1))
        plt.title("System packets Over Time")


        plt.tight_layout()
        plt.show()

    return area_system / current_time

replications = 30
averages = []

for j in range(replications):
    events_list = compute_event_queue(arrival_rate, departure_rate, 50)
    averages.append(simulate_with_list(events_list))

p = arrival_rate/departure_rate
print(f"Avg empirical {np.mean(averages):.3f}")
print(f"Avg theoretical {p/(1-p):.3f}")

def simulate_with_no_list(arrival_rate, departure_rate):
    current_time = 0
    index_event = 0

    server_status = 0
    in_queue = 0
    last_event_time = 0

    area_server = 0
    area_queue = 0

    server_status_history = [0]
    queue_length_history = [0]
    times = [0]
    system_history = [0]
    area_system = 0

    events_list = [(np.random.exponential(1 / arrival_rate), 'a')]

    while True:
        event = events_list.pop(0)
        print(f"Current event is \'{event[1]}\' at time {event[0]}")

        current_time = event[0]
        type_event = event[1]

        
        area_server += (current_time - last_event_time) * server_status
        area_queue += in_queue * (current_time - last_event_time)
        area_system += in_queue * (current_time - last_event_time) + (current_time - last_event_time) * server_status


        if type_event == 'a':
            next_arrival = (current_time + np.random.exponential(1 / arrival_rate), 'a')
            events_list.append(next_arrival)

            events_list = sorted(events_list, key=lambda x: x[0])

            if server_status == 0:
                server_status = 1
                service_time = np.random.exponential(1 / departure_rate)
                next_departure = (current_time + service_time, 'd')
                events_list.append(next_departure)
                events_list = sorted(events_list, key=lambda x: x[0])
            else:
                in_queue += 1
        else:
            
            server_status = 0

            if in_queue != 0:
                server_status = 1
                service_time = np.random.exponential(1 / departure_rate)
                next_departure = (current_time + service_time, 'd')
                events_list.append(next_departure)
                events_list = sorted(events_list, key=lambda x: x[0])
                in_queue -= 1

        print(f"Area Q(t) {area_queue:.3f}, Area U(t) {area_server:.3f}, Area system {area_system:3f}")
        
        times.append(current_time)
        server_status_history.append(server_status)
        queue_length_history.append(in_queue)
        system_history.append(server_status+in_queue)

        last_event_time = current_time

        p = arrival_rate/departure_rate
        print(f"Average number of packets in the system {(p*(1-p)):.3f}")
        print(f"Empirical number of packets in the system {(area_system / current_time):.3f}")
        print(f"Queue length {in_queue}")

        user_input = input("Premi Invio per continuare, qualsiasi altro tasto + Invio per fermare: \n")
        if user_input != "":
            break


    #finishing last events scheduled
    while len(events_list) != 0:
        event = events_list.pop(0)
        print(f"Current event is \'{event[1]}\' at time {event[0]}")

        current_time = event[0]
        type_event = event[1]

        
        area_server += (current_time - last_event_time) * server_status
        area_queue += in_queue * (current_time - last_event_time)
        area_system += in_queue * (current_time - last_event_time) + (current_time - last_event_time) * server_status


        if type_event == 'a':

            if server_status == 0:
                service_time = np.random.exponential(1 / departure_rate)
                next_departure = (current_time + service_time, 'd')
                events_list.append(next_departure)
                events_list = sorted(events_list, key=lambda x: x[0])
                server_status = 1
            else:
                in_queue += 1
        else:
            
            server_status = 0

            if in_queue != 0:
                server_status = 1
                service_time = np.random.exponential(1 / departure_rate)
                next_departure = (current_time + service_time, 'd')
                events_list.append(next_departure)
                events_list = sorted(events_list, key=lambda x: x[0])
                in_queue -= 1

        print(f"Area Q(t) {area_queue:.3f}, Area U(t) {area_server:.3f}, Area system {area_system:3f}")
        
        times.append(current_time)
        server_status_history.append(server_status)
        queue_length_history.append(in_queue)
        system_history.append(server_status+in_queue)

        last_event_time = current_time

        p = arrival_rate/departure_rate
        print(f"Average number of packets in the system {(p*(1-p)):.3f}")
        print(f"Empirical number of packets in the system {(area_system / current_time):.3f}")
        print(f"Queue length {in_queue}\n")


    max_y = max(max(system_history), max(queue_length_history), max(server_status_history))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.plot(times, server_status_history, drawstyle='steps-post')
    plt.xlabel("Time")
    plt.ylabel("Server status")
    plt.yticks(np.arange(0, 3, 1)) 
    plt.title("Server Status Over Time")

    plt.subplot(1, 3, 2)
    plt.plot(times, queue_length_history, drawstyle='steps-post', color='orange')
    plt.xlabel("Time")
    plt.ylabel("Queue length")
    plt.yticks(np.arange(0, max_y+2, 1))
    plt.title("Queue Length Over Time")

    plt.subplot(1, 3, 3)
    plt.plot(times, system_history, drawstyle='steps-post', color='green')
    plt.xlabel("Time")
    plt.ylabel("System packets")
    plt.yticks(np.arange(0, max_y+2, 1))
    plt.title("System packets Over Time")


    plt.tight_layout()
    plt.show()

#simulate_with_no_list(2, 3)