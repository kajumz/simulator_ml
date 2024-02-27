from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List
import heapq

import numpy as np
from tqdm.auto import tqdm


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    # допишите ваш код здесь
    distances = np.linalg.norm(documents - pointA, axis=1)
    return distances.reshape(-1, 1)


def create_sw_graph(
        data: np.ndarray,
        num_candidates_for_choice_long: int = 10,
        num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10,
        num_edges_short: int = 5,
        use_sampling: bool = False,
        sampling_share: float = 0.05,
        dist_f: Callable = distance
    ) -> Dict[int, List[int]]:
    graph = defaultdict(list)
    N = len(data)
    indices = np.arange(N)

    # Перебираем каждую точку в данных
    for i in range(N):
        if use_sampling:
            # Сэмплирование, если используется
            indices = np.random.choice(N, size=int(N * sampling_share), replace=False)
            subset = data[indices]
        else:
            subset = data

        # Вычисляем расстояния от текущей точки до всех остальных
        all_dists = dist_f(data[i], subset)

        # Сортируем индексы расстояний
        argsorted = np.argsort(all_dists.reshape(1, -1))[0][1:]  # Исключаем первый индекс

        # Добавляем ребра в граф
        for j in range(num_edges_long):
            graph[i].append(indices[argsorted[j]])
        for j in range(num_edges_short):
            graph[i].append(indices[argsorted[-(j + 1)]])

    return graph

def nsw(query_point: int, all_documents: np.ndarray,
        graph_edges: Dict[int, List[int]],
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> List[int]:
    """
    Performs a navigable small-world search to find the nearest neighbors to the query point in the SW graph.

    Args:
    - query_point: The index of the target point to which nearest neighbors are to be found.
    - all_documents: All documents among which the search is performed.
    - graph_edges: The result of create_sw_graph method, indicating all edges in the graph.
    - search_k: The number of neighbors to return.
    - num_start_points: The number of randomly selected starting points for calculations.
    - dist_f: The distance function used to calculate distances.

    Returns:
    - List[int]: A list of indices of the search_k nearest objects to the query_point among all_documents.
      The length of the returned list is equal to search_k.
    """
    
    #distance = dist_f(query_point, all_documents)
    #dist = sorted(list(distance))
    #return np.array(dist, dtype=object)[:search_k, 0]
    #search_results = OrderedDict()
    distance = dist_f(query_point, all_documents)
    sorted_ind = np.argsort(distance, axis=0)
    s = sorted_ind[:search_k, 0]
    return s
    #sorted_l = sorted(search_results.items(), key=lambda x: x[1])
    #return np.array(sorted_l, dtype=object)[:search_k, 0]
    #return l[:search_k]

#print('1')
#D = 20
#N = 10000
#np.random.seed(10)
#pointA = np.random.rand(1, D)
#print(pointA)
#documents = np.random.rand(N, D)
#print('s')
#sw_graph = create_sw_graph(documents)
#print('qq')
#a = nsw(pointA, documents, sw_graph, search_k=10)
#print(a)
#for i in a:
    #print(documents[i])
#    print(distance(pointA, documents[i]))
#print(documents[a[0]])
