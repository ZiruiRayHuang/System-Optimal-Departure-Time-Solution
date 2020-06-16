import datetime
import os
import sys
import time as tm
import pickle
from tqdm import tqdm
#---------------------------- DEFINE FUNCTIONS AND CLASSES-----------------------------------
class FibonacciHeap:

    # internal node class
    class Node:
        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.parent = self.child = self.left = self.right = None
            self.degree = 0
            self.mark = False

    # function to iterate through a doubly linked list
    def iterate(self, head):
        node = stop = head
        flag = False
        while True:
            if node == stop and flag is True:
                break
            elif node == stop:
                flag = True
            yield node
            node = node.right

    # pointer to the head and minimum node in the root list
    root_list, min_node = None, None

    # maintain total node count in full fibonacci heap
    total_nodes = 0

    # return min node in O(1) time
    def find_min(self):
        return self.min_node

    # extract (delete) the min node from the heap in O(log n) time
    # amortized cost analysis can be found here (http://bit.ly/1ow1Clm)
    def extract_min(self):
        z = self.min_node
        if z is not None:
            if z.child is not None:
                # attach child nodes to root list
                children = [x for x in self.iterate(z.child)]
                for i in range(0, len(children)):
                    self.merge_with_root_list(children[i])
                    children[i].parent = None
            self.remove_from_root_list(z)
            # set new min node in heap
            if z == z.right:
                self.min_node = self.root_list = None
            else:
                self.min_node = z.right
                self.consolidate()
            self.total_nodes -= 1
        return z

    # insert new node into the unordered root list in O(1) time
    def insert(self, key, value=None):
        n = self.Node(key, value)
        n.left = n.right = n
        self.merge_with_root_list(n)
        if self.min_node is None or n.key < self.min_node.key:
            self.min_node = n
        self.total_nodes += 1
        return n

    # modify the key of some node in the heap in O(1) time
    def decrease_key(self, x, k):
        if k > x.key:
            return None
        x.key = k
        y = x.parent
        if y is not None and x.key < y.key:
            self.cut(x, y)
            self.cascading_cut(y)
        if x.key < self.min_node.key:
            self.min_node = x

    # merge two fibonacci heaps in O(1) time by concatenating the root lists
    # the root of the new root list becomes equal to the first list and the second
    # list is simply appended to the end (then the proper min node is determined)
    def merge(self, h2):
        H = FibonacciHeap()
        H.root_list, H.min_node = self.root_list, self.min_node
        # fix pointers when merging the two heaps
        last = h2.root_list.left
        h2.root_list.left = H.root_list.left
        H.root_list.left.right = h2.root_list
        H.root_list.left = last
        H.root_list.left.right = H.root_list
        # update min node if needed
        if h2.min_node.key < H.min_node.key:
            H.min_node = h2.min_node
        # update total nodes
        H.total_nodes = self.total_nodes + h2.total_nodes
        return H

    # if a child node becomes smaller than its parent node we
    # cut this child node off and bring it up to the root list
    def cut(self, x, y):
        self.remove_from_child_list(y, x)
        y.degree -= 1
        self.merge_with_root_list(x)
        x.parent = None
        x.mark = False

    # cascading cut of parent node to obtain good time bounds
    def cascading_cut(self, y):
        z = y.parent
        if z is not None:
            if y.mark is False:
                y.mark = True
            else:
                self.cut(y, z)
                self.cascading_cut(z)

    # combine root nodes of equal degree to consolidate the heap
    # by creating a list of unordered binomial trees
    def consolidate(self):
        A = [None] * self.total_nodes
        nodes = [w for w in self.iterate(self.root_list)]
        for w in range(0, len(nodes)):
            x = nodes[w]
            d = x.degree
            while A[d] != None:
                y = A[d]
                if x.key > y.key:
                    temp = x
                    x, y = y, temp
                self.heap_link(y, x)
                A[d] = None
                d += 1
            A[d] = x
        # find new min node - no need to reconstruct new root list below
        # because root list was iteratively changing as we were moving
        # nodes around in the above loop
        for i in range(0, len(A)):
            if A[i] is not None:
                if A[i].key < self.min_node.key:
                    self.min_node = A[i]

    # actual linking of one node to another in the root list
    # while also updating the child linked list
    def heap_link(self, y, x):
        self.remove_from_root_list(y)
        y.left = y.right = y
        self.merge_with_child_list(x, y)
        x.degree += 1
        y.parent = x
        y.mark = False

    # merge a node with the doubly linked root list
    def merge_with_root_list(self, node):
        if self.root_list is None:
            self.root_list = node
        else:
            node.right = self.root_list.right
            node.left = self.root_list
            self.root_list.right.left = node
            self.root_list.right = node

    # merge a node with the doubly linked child list of a root node
    def merge_with_child_list(self, parent, node):
        if parent.child is None:
            parent.child = node
        else:
            node.right = parent.child.right
            node.left = parent.child
            parent.child.right.left = node
            parent.child.right = node

    # remove a node from the doubly linked root list
    def remove_from_root_list(self, node):
        if node == self.root_list:
            self.root_list = node.right
        node.left.right = node.right
        node.right.left = node.left

    # remove a node from the doubly linked child list
    def remove_from_child_list(self, parent, node):
        if parent.child == parent.child.right:
            parent.child = None
        elif parent.child == node:
            parent.child = node.right
            node.right.parent = parent
        node.left.right = node.right
        node.right.left = node.left

def dijkstra(adjList, source, sink = None, dist_data = None):
    n = len(adjList)    #intentionally 1 more than the number of vertices, keep the 0th entry free for convenience
    visited = [False]*n
    distance = [float('inf')]*n
    previous = [None]*n

    heapNodes = [None]*n
    heap = FibonacciHeap()
    for i in range(1, n):
        heapNodes[i] = heap.insert(float('inf'), i)     # distance, label

    distance[source] = 0
    previous[source] = 0
    heap.decrease_key(heapNodes[source], 0)

    while heap.total_nodes:
        current = heap.extract_min().value
        distance[current] = distance[current] - dist_data[current - 1][sink - 1]
        visited[current] = True

        #early exit
        if sink and current == sink:
            break

        for (neighbor, cost) in adjList[current]:
            if not visited[neighbor]:
                if distance[current] + cost < distance[neighbor]:
                    distance[neighbor] = distance[current] + cost + dist_data[neighbor - 1][sink - 1]
                    previous[neighbor] = current
                    heap.decrease_key(heapNodes[neighbor], distance[neighbor])

    path = []
    cost = []
    next_node = sink

    while True:
        path = [next_node] + path
        cost = [distance[next_node]] + cost
        next_node = previous[next_node]
        if next_node == source:
            path = [source] + path
            cost = [0] + cost
            break

    cost = [int(cost[i] / 30) for i in range(len(cost))]

    return path,cost


def round_to_1000(x, base=1000):
    return base * round(x/base) if round(x/base) !=0 else base

def calculate_score(vol_data, capacity_data, start_time, end_time):
    score_sum = 0
    for i in range(int(start_time / 30), int(end_time / 30)):
        score_sum = score_sum + sum([pow(vol_data[i][j] * 2 / capacity_data[j],2) for j in range(len(capacity_data))])
    return score_sum

def Dijkstra(time_interval, adjList, source, sink, dist_data):
    path, cost = dijkstra(adjList, source, sink, dist_data)
    cost = [cost[i] + time_interval for i in range(len(cost))]
    return path, cost


#------------------------------------ RUNNING ALGORITHM  -----------------------------------
pardir = os.path.abspath(os.path.join(os.path.realpath(sys.executable),os.path.pardir,os.path.pardir))
if os.path.exists(pardir + r'\Output\\'):
    pass
else:
    os.mkdir(pardir + r'\Output\\')

print()
print()
print()
print('******************* PART II: RUNNING ALGORITHM *****************')
print()

global dist_data, t_0_data, capacity_data, link_index_adjList_mapping_i, link_index_adjList_mapping_j, link_index_adjList_mapping_fromid, link_index_adjList_mapping_toid, link_data_index

with open(pardir + r'\Preprocessed\dist_data.dat','rb') as filehandle:
    dist_data = pickle.load(filehandle)

with open(pardir + r'\Preprocessed\t_0_data.dat', 'rb') as filehandle:
    t_0_data = pickle.load(filehandle)
with open(pardir + r'\Preprocessed\capacity_data.dat', 'rb') as filehandle:
    capacity_data = pickle.load(filehandle)

capacity_data_square = [pow(i, 2) for i in capacity_data]

with open(pardir + r'\Preprocessed\link_index_adjList_mapping_i.dat', 'rb') as filehandle:
    link_index_adjList_mapping_i = pickle.load(filehandle)

with open(pardir + r'\Preprocessed\link_index_adjList_mapping_j.dat', 'rb') as filehandle:
    link_index_adjList_mapping_j = pickle.load(filehandle)

with open(pardir + r'\Preprocessed\link_index_adjList_mapping_fromid.dat', 'rb') as filehandle:
    link_index_adjList_mapping_fromid = pickle.load(filehandle)

with open(pardir + r'\Preprocessed\link_index_adjList_mapping_toid.dat', 'rb') as filehandle:
    link_index_adjList_mapping_toid = pickle.load(filehandle)

with open(pardir + r'\Preprocessed\link_data_index.dat', 'rb') as filehandle:
    link_data_index = pickle.load(filehandle)

with open(pardir + r'\Preprocessed\vol_data.dat', 'rb') as filehandle:
    vol_data = pickle.load(filehandle)

with open(pardir + r'\Preprocessed\vehicle_data_Stime_30min', 'rb') as filehandle:
    vehicle_data_Stime_30min = pickle.load(filehandle)

with open(pardir + r'\Preprocessed\vehicle_data_participate', 'rb') as filehandle:
    vehicle_data_participate = pickle.load(filehandle)

with open(pardir + r'\Preprocessed\vehicle_data_tolerance', 'rb') as filehandle:
    vehicle_data_tolerance = pickle.load(filehandle)

with open(pardir + r'\Preprocessed\path_data.dat', 'rb') as filehandle:
    path_data = pickle.load(filehandle)

with open(pardir + r'\Preprocessed\adjListAll.dat', 'rb') as filehandle:
    adjListAll = pickle.load(filehandle)

path = pardir + r'\Input\config.dat'
with open(path, 'r') as file:
    data = file.readlines()
start_time = int(data[0].rstrip().split()[0])
end_time = int(data[0].rstrip().split()[1])
peak = [[] for _ in range(len(data) - 3)]
for i in range(len(data) - 3):
    peak[i] = [int(data[1 + i].rstrip().split()[0]), int(data[1 + i].rstrip().split()[1])]
participate_rate = float(data[-2].rstrip()) / 100
max_permissive_hour = int(data[-1].rstrip())

alpha = 0.15
beta = 4.0

shift_data = [[] for _ in range(len(path_data))]
time_data = [[] for _ in range(len(path_data))]

initial_score = calculate_score(vol_data, capacity_data, start_time, end_time)
score = initial_score

vehicle_participate = [x for x,y in enumerate(vehicle_data_participate) if y == 1]

print('Participate Rate: %s  Length of Permissive Hour: %s' %(str(participate_rate), str(max_permissive_hour)))
print()
print('# of Vehicles To Be Adjusted: %s ' %(str(len(vehicle_participate))))
print()
print('Start Time: %s ' %(datetime.datetime.now().replace(microsecond=0)))
print()

for iteration in range(1):
    #print('----------------------- Iteration <%s/3> ------------------------' %(str(iteration + 1)))
    num_vehicle = 0
    for index in tqdm(vehicle_participate[0:302]):
        num_vehicle = num_vehicle + 1
        usec = path_data[index][0]
        dsec = s_node = path_data[index][1]
        t_node = path_data[index][-1]
        if usec == t_node:
            continue
        if s_node == t_node:
            continue
        time_interval = int(vehicle_data_Stime_30min[index])
        min_score = score
        min_vol = [vol_data[i][:] for i in range(int(start_time / 30), int(end_time / 30))]

        if iteration == 0:
            min_shift = 0
            min_path, min_time = Dijkstra(time_interval, adjListAll[time_interval], s_node, t_node, dist_data)
            min_path = [usec] + min_path
        else:
            min_shift = shift_data[index][iteration - 1][0]
            min_path = path_data[index]
            min_time = time_data[index]

        score_minus = 0
        for i in range(0,len(min_path) - 1):
            link_index = link_data_index[min_path[i] - 1][min_path[i + 1] - 1]
            time_index = min_time[i]
            if time_index >= int(end_time / 30):
                break
            vol_data[time_index][link_index] = vol_data[time_index][link_index] - 1
            index_1 = link_index_adjList_mapping_i[link_index]
            index_2 = link_index_adjList_mapping_j[link_index]
            adjListAll[time_index][index_1][index_2][1] = t_0_data[link_index] * (1 + alpha * pow(vol_data[time_index][link_index] * 2 / capacity_data[link_index], beta))
            score_minus = score_minus + (vol_data[time_index][link_index] * 4 + 2) * 2 / capacity_data_square[link_index]
        score_0 = min_score - score_minus

        for shift in range(-1 * int(vehicle_data_tolerance[index]), 1 * int(vehicle_data_tolerance[index]) + 1):
            if shift == min_shift:
                continue
            if time_interval + shift >= int(end_time / 30):
                continue
            if time_interval + shift < 0:
                continue
            tempory_vol = [vol_data[i][:] for i in range(int(start_time / 30), int(end_time / 30))]
            path, time = Dijkstra(time_interval + shift, adjListAll[time_interval + shift], s_node, t_node, dist_data)
            path = [usec] + path
            score_add = 0
            for i in range(len(path) - 1):
                link_index = link_data_index[path[i] - 1][path[i + 1] - 1]
                time_index = time[i]
                if time_index >= int(end_time / 30):
                    break
                tempory_vol[time_index][link_index] = tempory_vol[time_index][link_index] + 1
                score_add = score_add + (tempory_vol[time_index][link_index] * 4 - 2) * 2 / capacity_data_square[link_index]
            if score_0 + score_add < min_score:
                min_vol = [tempory_vol[i][:] for i in range(int(start_time / 30), int(end_time / 30))]
                min_score = score_0 + score_add
                min_shift = shift
                min_time = time
                min_path = path
        vol_data = [min_vol[i][:] for i in range(int(start_time / 30), int(end_time / 30))]
        score = min_score
        path_data[index] = min_path
        time_data[index] = min_time
        shift_data[index].append([min_shift, min_score])

        for i in range(0,len(min_path) - 1):
            link_index = link_data_index[min_path[i] - 1][min_path[i + 1] - 1]
            time_index = min_time[i]
            if time_index >= int(end_time / 30):
                break
            index_1 = link_index_adjList_mapping_i[link_index]
            index_2 = link_index_adjList_mapping_j[link_index]
            adjListAll[time_index][index_1][index_2][1] = t_0_data[link_index] * (1 + alpha * pow(vol_data[time_index][link_index] * 2 / capacity_data[link_index], beta))

        #if (num_vehicle) % round_to_1000(len(vehicle_participate) / 10) == 0:
            #print('# of Travelers Adjusted: %s   Current Time: %s ' % (str(num_vehicle), datetime.datetime.now().replace(microsecond=0)),flush=True)

tm.sleep(0.5)
print()

print('Saving Files ...  ', end='',flush=True)

with open(pardir + r'\Output\shift_data.dat', 'wb') as filehandle:
    pickle.dump(shift_data, filehandle)
with open(pardir + r'\Output\path_data.dat', 'wb') as filehandle:
    pickle.dump(path_data, filehandle)

print('Done')