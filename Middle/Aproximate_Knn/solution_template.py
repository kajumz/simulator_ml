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

def nsw(query_point: np.ndarray, all_documents: np.ndarray, 
        graph_edges: Dict[int, List[int]],
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    def insert_into_ordered_dict(ordered_dict, key, value, priority):
        if key in ordered_dict:
            ordered_dict[key].append((priority, value))
        else:
            ordered_dict[key] = [(priority, value)]

    def search_neighbors(current_point, visited, ordered_dict):
        for neighbor in graph_edges[current_point]:
            if neighbor not in visited:
                priority = dist_f(all_documents[neighbor], query_point)
                insert_into_ordered_dict(ordered_dict, neighbor, neighbor, priority)

    search_results = OrderedDict()
    visited = set()
    priority_queue = OrderedDict()

    # Select initial points to start the search
    start_points = np.random.choice(len(all_documents), size=num_start_points, replace=False)
    for start_point in start_points:
        priority = dist_f(all_documents[start_point], query_point)
        insert_into_ordered_dict(priority_queue, start_point, start_point, priority)

    while len(search_results) < search_k and priority_queue:
        current_point = next(iter(priority_queue))
        del priority_queue[current_point]
        visited.add(current_point)
        search_results[current_point] = dist_f(all_documents[current_point], query_point)
        search_neighbors(current_point, visited, priority_queue)

    return np.array(list(search_results.items()))[:search_k, 0]


