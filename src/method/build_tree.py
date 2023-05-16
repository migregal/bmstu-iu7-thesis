from deduplicate import find_closest_intersection
from utils import get_intersection

if __name__ == "__main__":
    # bboxes = [[-2, 2, 2, 6], [-1, 3, 3, 7], [0, 3, 2, 5]]
    # bboxes = [[-2, 2, 2, 6], [-1, 3, 3, 7], [0, 3, 2, 5], [3, 10, 5, 8]]
    # bboxes = [[0.42022, 0.66965, 0.52942, 0.85779], [0.38847, 0.18751, 0.4746, 0.3812]]
    # bboxes = [[4, 6, 6, 9], [3.5, 1.5, 5, 4]]
    # print(get_intersection(*bboxes))

    bboxes = [
        [0.41777, 0.67981, 0.53591, 0.86522],
        [0.42028, 0.6638, 0.53469, 0.86325],
        [0.42022, 0.66965, 0.52942, 0.85779],
    ]
    print(find_closest_intersection(bboxes))
