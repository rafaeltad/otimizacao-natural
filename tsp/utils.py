import geopy.distance

def get_distance(coords1: tuple, coords2: tuple):
    return geopy.distance.geodesic(coords1, coords2).km